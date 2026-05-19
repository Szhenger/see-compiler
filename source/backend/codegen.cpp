#include "include/backend/codegen.hpp"
#include "include/utility/logger.hpp"

#include <format>

namespace seecpp::backend {

// =============================================================================
// CodegenDriver::Constructor & Pass Management
// =============================================================================

CodegenDriver::CodegenDriver(std::unique_ptr<ICodegenTarget> target, CodegenOptions options)
    : target_(std::move(target)), options_(std::move(options)) {}

void CodegenDriver::add_pass(std::unique_ptr<IBackendPass> pass) {
    passes_.push_back(std::move(pass));
}

// =============================================================================
// CodegenDriver::allocateBuffers (The Linear Memory Mapper)
// =============================================================================

std::expected<AllocationResult, BackendError>
CodegenDriver::allocateBuffers(const sir::Block& block, const utility::WeightBuffer& weights) {
    AllocationResult result;
    size_t watermark = 0;
    const size_t alignment = options_.required_alignment;

    // Helper lambda to DRY up the allocation logic
    auto assign_slot = [&](const sir::Value* v) {
        if (result.allocation.contains(v)) return;

        BufferSlot slot;
        const std::string vid(v->id());

        // Weights: Zero-offset, managed by the external WeightBuffer
        if (weights.contains(vid)) {
            slot.is_weight = true;
            slot.offset    = 0; 
            slot.byte_size = v->shape().byteSize(v->dtype());
            slot.alignment = alignment;
        } 
        // Activations: Assigned to the continuous Memory Arena
        else {
            const size_t sz = v->shape().byteSize(v->dtype());
            
            // 1. Force the watermark forward to the nearest hardware-aligned boundary.
            // Formula: (watermark + align - 1) & ~(align - 1) requires align to be a power of 2.
            watermark = (watermark + alignment - 1) & ~(alignment - 1);

            slot.is_weight = false;
            slot.offset    = watermark;
            slot.byte_size = sz;
            slot.alignment = alignment;

            // 2. Advance the watermark by the tensor size
            watermark += sz;
        }

        result.allocation.emplace(v, slot);
    };

    // 1. Map all graph inputs (block arguments) to the arena
    for (const auto& arg : block.arguments()) {
        assign_slot(arg.get());
    }

    // 2. Map all operation outputs in forward topological order
    block.walk([&](const sir::Operation* op) {
        for (size_t i = 0; i < op->numResults(); ++i) {
            assign_slot(op->result(i));
        }
    });

    // 3. Final alignment pad for the total Arena struct size
    result.arena_bytes = (watermark + alignment - 1) & ~(alignment - 1);

    utility::Logger::info(std::format(
        "BufferAllocator: {} value(s) statically mapped. Total Arena Size: {} bytes (Aligned to {}B)",
        result.allocation.size(), result.arena_bytes, alignment
    ));

    return result;
}

// =============================================================================
// CodegenDriver::run (The Pipeline Orchestrator)
// =============================================================================

std::expected<void, BackendError>
CodegenDriver::run(sir::Block& block, utility::WeightBuffer& weights) {
    
    utility::Logger::info(std::format(
        "CodegenDriver: Initiating pipeline targeting '{}'.", target_->name()
    ));

    // --- Stage 1: Dynamic Pre-Codegen Passes ---
    // Sequentially executes any registered mutation passes (e.g., WeightFolding, Quantization)
    for (const auto& pass : passes_) {
        utility::Logger::info(std::format("CodegenDriver: Executing pass '{}'", pass->name()));
        
        if (auto res = pass->run(block, weights, options_); !res) {
            return std::unexpected(res.error());
        }
    }

    // --- Stage 2: Memory Binding (Arena Allocation) ---
    auto alloc_result = allocateBuffers(block, weights);
    if (!alloc_result) {
        return std::unexpected(alloc_result.error());
    }
    
    // Cache the result state for querying after compilation
    latest_allocation_ = *alloc_result;

    // --- Stage 3: Hardware Target Emission ---
    utility::Logger::info("CodegenDriver: Entering target emission phase.");
    
    if (auto res = target_->emit(block, latest_allocation_, weights, options_); !res) {
        return std::unexpected(res.error());
    }

    utility::Logger::info(std::format(
        "CodegenDriver: Pipeline execution successful. Emitted artifacts for '{}'.", 
        target_->name()
    ));

    return {};
}

} // namespace seecpp::backend
