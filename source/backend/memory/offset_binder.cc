#include "source/memory/offset_binder.h"
#include "include/utility/logger.h"

namespace seecpp::backend {

std::expected<uint64_t, CodegenError> OffsetBinder::Run(sir::Block& block) {
    size_t watermark = 0;
    
    // AVX-512 requires 64-byte alignment and 64-byte vector loads
    const size_t kVectorWidthBytes = 64; 

    std::unordered_set<const sir::Value*> mapped_values;

    auto assign_slot = [&](sir::Value* v, sir::Operation* producer_op) {
        if (mapped_values.contains(v)) return;

        const size_t actual_bytes = v->shape().byteSize(v->dtype());
        
        // SECURE PADDING FIX: 
        // Force the tensor's footprint to perfectly encompass full SIMD vectors.
        // This guarantees a kernel can do a blind 64-byte store without corrupting 
        // the adjacent tensor.
        const size_t padded_bytes = (actual_bytes + kVectorWidthBytes - 1) & ~(kVectorWidthBytes - 1);

        // 1. Align the starting offset
        watermark = (watermark + kVectorWidthBytes - 1) & ~(kVectorWidthBytes - 1);

        // 2. Bind the offset
        if (producer_op) {
            auto current_offsets = producer_op->GetAttribute<std::vector<int64_t>>("output_offsets")
                                              .value_or(std::vector<int64_t>{});
            current_offsets.push_back(static_cast<int64_t>(watermark));
            producer_op->SetAttribute("output_offsets", current_offsets);
        }

        // 3. Advance the watermark by the PADDED size, not the actual size.
        watermark += padded_bytes;
        
        mapped_values.insert(v);
    };

    // ... [Rest of the AST walk remains identical] ...

    // 4. Final Arena Padding
    // Because every tensor was advanced by a multiple of 64, the watermark is 
    // already guaranteed to be a safe multiple of 64 bytes. We do one final 
    // safety align just to be absolutely certain before returning.
    const uint64_t safe_arena_bytes = (watermark + kVectorWidthBytes - 1) & ~(kVectorWidthBytes - 1);

    utility::Logger::Info(std::format(
        "OffsetBinder: Arena Secured. {} mapped. Total Size: {} bytes.",
        mapped_values.size(), safe_arena_bytes
    ));

    return safe_arena_bytes;
}

} // namespace seecpp::backend
