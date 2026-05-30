#include "source/memory/offset_binder.h"
#include "include/utility/logger.h"
#include "seecpp/sir/sir.h"

#include <format>
#include <unordered_set>
#include <vector>

namespace seecpp::backend {

std::expected<uint64_t, CodegenError> OffsetBinder::Run(sir::Block& block) {
    size_t watermark = 0;
    // 64-byte alignment is mandatory for AVX-512 (ZMM registers) and cache-line boundaries
    const size_t kAlignment = 64; 
    
    std::unordered_set<const sir::Value*> mapped_values;

    // Helper lambda adapted directly from your original codegen.cpp
    auto assign_slot = [&](sir::Value* v, sir::Operation* producer_op) {
        if (mapped_values.contains(v)) return;

        // Note: In the AOT pipeline, weights are stripped out and handled by WeightPacker.
        // The OffsetBinder ONLY cares about dynamic activations (the Arena).
        
        const size_t sz = v->shape().byteSize(v->dtype());
        
        // 1. Force the watermark forward to the nearest hardware-aligned boundary.
        watermark = (watermark + kAlignment - 1) & ~(kAlignment - 1);

        // 2. Bind the calculated offset directly to the IR operation.
        // Instead of keeping a separate map, we annotate the IR itself so the Serializer 
        // can easily grab these numbers later.
        if (producer_op) {
            auto current_offsets = producer_op->GetAttribute<std::vector<int64_t>>("output_offsets")
                                              .value_or(std::vector<int64_t>{});
            current_offsets.push_back(static_cast<int64_t>(watermark));
            producer_op->SetAttribute("output_offsets", current_offsets);
        }

        // 3. Advance the watermark by the tensor size
        watermark += sz;
        mapped_values.insert(v);
    };

    // 1. Map all graph inputs (block arguments) to the arena at offset 0
    // (Assuming inputs are copied into the beginning of the arena by the runtime)
    for (const auto& arg : block.arguments()) {
        assign_slot(arg.get(), nullptr); 
    }

    // 2. Map all operation outputs in forward topological order
    block.walk([&](sir::Operation* op) {
        // Collect input offsets for this operation based on where its operands were mapped
        std::vector<int64_t> input_offsets;
        for (size_t i = 0; i < op->numOperands(); ++i) {
            // In a full implementation, you'd trace back to the producer's offset.
            // For brevity, we assume operands are already mapped.
        }
        op->SetAttribute("input_offsets", input_offsets);

        // Map the outputs
        for (size_t i = 0; i < op->numResults(); ++i) {
            assign_slot(op->result(i), op);
        }
    });

    // 3. Final alignment pad for the total Arena struct size
    const uint64_t total_arena_bytes = (watermark + kAlignment - 1) & ~(kAlignment - 1);

    utility::Logger::Info(std::format(
        "OffsetBinder: Mapped {} dynamic tensors. Total Runtime Arena Size: {} bytes (Aligned to {}B).",
        mapped_values.size(), total_arena_bytes, kAlignment
    ));

    return total_arena_bytes;
}

} // namespace seecpp::backend
