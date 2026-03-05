#include "include/backend/codegen.hpp"
#include "include/utility/logger.hpp"

namespace seecpp::backend {

// =============================================================================
// BufferAllocator::allocate
//
// Linear-scan allocation strategy:
//
//   For each op in forward order:
//     For each result value:
//       If the value is a weight (its id exists in WeightBuffer):
//         Mark is_weight=true, offset=0 (WeightBuffer manages its own memory).
//       Else:
//         Assign arena slot at current watermark.
//         Advance watermark by value.shape().byteSize(dtype).
//
// This is a conservative strategy — no lifetime analysis, no slot reuse.
// Every activation lives in the arena for the full duration of the block.
// A future graph-colouring allocator can reuse slots for values whose
// lifetimes do not overlap (standard register allocation literature).
//
// All block arguments (graph inputs) are also assigned slots so the
// codegen target can write the input tensor addresses into the arena.
// =============================================================================

std::expected<BufferAllocator::Result, CodegenError>
BufferAllocator::allocate(const sir::Block&            block,
                           const utility::WeightBuffer& weights) {
    Result result;
    size_t watermark = 0;

    // Allocate slots for block arguments (graph inputs).
    for (const auto& arg : block.arguments()) {
        assignSlot(arg.get(), result.allocation, watermark, weights);
    }

    // Allocate slots for all op results in forward order.
    block.walk([&](const sir::Operation* op) {
        for (size_t i = 0; i < op->numResults(); ++i) {
            const sir::Value* v = op->result(i);
            assignSlot(v, result.allocation, watermark, weights);
        }
    });

    result.arena_bytes = watermark;

    utility::Logger::info(
        "BufferAllocator: " +
        std::to_string(result.allocation.size()) + " value(s) allocated, " +
        "arena=" + std::to_string(watermark) + " bytes");

    return result;
}

BufferSlot& BufferAllocator::assignSlot(const sir::Value*            v,
                                         BufferAllocation&             alloc,
                                         size_t&                       watermark,
                                         const utility::WeightBuffer&  weights) {
    // Already allocated (can happen if the same value is visited twice).
    auto it = alloc.find(v);
    if (it != alloc.end())
        return it->second;

    BufferSlot slot;

    // Weight values (produced by sc_high.constant) live in WeightBuffer.
    // Check by id: if WeightBuffer contains this key, it's a weight.
    const std::string vid(v->id());
    if (weights.contains(vid)) {
        slot.is_weight = true;
        slot.offset    = 0;  // WeightBuffer manages its own contiguous blob.
        slot.byte_size = v->shape().byteSize(v->dtype());
    } else {
        // Activation value: carve a slot from the arena.
        const size_t sz = v->shape().byteSize(v->dtype());
        slot.is_weight  = false;
        slot.offset     = watermark;
        slot.byte_size  = sz;

        // Align to 64 bytes for SIMD-friendly access.
        // This matches the alignment expected by AVX-512 intrinsics and
        // cuBLAS device pointers.
        constexpr size_t kAlign = 64;
        watermark += sz;
        watermark  = (watermark + kAlign - 1) & ~(kAlign - 1);
    }

    alloc.emplace(v, slot);
    return alloc.at(v);
}

// =============================================================================
// CodegenDriver::run
// =============================================================================

std::expected<void, CodegenError>
CodegenDriver::run(sir::Block& block, utility::WeightBuffer& weights) {

    utility::Logger::info(
        std::string("CodegenDriver: starting pipeline -> target='") +
        std::string(target_->name()) + "'");

    // --- Stage 1: Weight Folding ---
    // Performs Conv+BN parameter arithmetic deferred by OperatorFusionPass.
    {
        WeightFoldingPass folder;
        if (auto res = folder.run(block, weights); !res)
            return std::unexpected(res.error());
    }

    // --- Stage 2: Buffer Allocation ---
    // Assigns arena offsets to every activation Value in sc_low.*.
    {
        BufferAllocator allocator;
        auto alloc_result = allocator.allocate(block, weights);
        if (!alloc_result)
            return std::unexpected(alloc_result.error());

        arena_bytes_ = alloc_result->arena_bytes;

        // --- Stage 3: Code Generation ---
        if (auto res = target_->emit(block, alloc_result->allocation, weights);
            !res)
            return std::unexpected(res.error());
    }

    utility::Logger::info(
        "CodegenDriver: pipeline complete — arena=" +
        std::to_string(arena_bytes_) + " bytes, target='" +
        std::string(target_->name()) + "'");

    return {};
}

} // namespace seecpp::codegen