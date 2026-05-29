#include "src/weights/weight_packer.h"
#include "include/utility/logger.h"

// Assuming your framework provides these
#include "seecpp/sir/sir.h"
#include "seecpp/utility/weight_buffer.h"

#include <format>
#include <span>

namespace seecpp::backend {

namespace {
// Helper function to calculate the padding required for memory alignment.
inline size_t CalculateAlignedOffset(size_t current_offset, size_t alignment) {
    return (current_offset + alignment - 1) & ~(alignment - 1);
}
}  // namespace

std::expected<PackedWeights, CodegenError> WeightPacker::Run(
    sir::Block& block, 
    const utility::WeightBuffer& weights,
    size_t alignment) 
{
    packed_tensor_count_ = 0;
    total_bytes_packed_ = 0;

    utility::Logger::Info(std::format(
        "WeightPacker: Starting .rodata compilation (Alignment: {} bytes)", 
        alignment
    ));

    PackedWeights result;
    // Pre-allocate a reasonable chunk to prevent continuous vector reallocations
    result.rodata_blob.reserve(1024 * 1024 * 10); // 10 MB initial reservation

    std::expected<void, CodegenError> pass_result = {};

    block.walk([&](sir::Operation* op) {
        if (!pass_result) return;

        // Iterate through all operands of the instruction
        for (size_t i = 0; i < op->numOperands(); ++i) {
            const std::string tensor_id = op->operand(i)->id();

            // 1. Skip if it's not a constant weight, or if we've already packed it
            if (!weights.has(tensor_id) || result.offsets.contains(tensor_id)) {
                continue;
            }

            // 2. Fetch the raw binary data from the WeightBuffer
            // (Assuming a method that returns std::span<const uint8_t>)
            auto byte_span = weights.getRawBytes(tensor_id);
            if (byte_span.empty()) {
                pass_result = std::unexpected(CodegenError{
                    "weight_packing",
                    std::format("WeightBuffer contains no data for tensor '{}'", tensor_id)
                });
                return;
            }

            // 3. Calculate alignment padding
            const size_t current_size = result.rodata_blob.size();
            const size_t aligned_offset = CalculateAlignedOffset(current_size, alignment);
            const size_t padding_bytes = aligned_offset - current_size;

            // 4. Inject zero-padding to reach the required hardware alignment
            if (padding_bytes > 0) {
                result.rodata_blob.insert(result.rodata_blob.end(), padding_bytes, 0x00);
            }

            // 5. Record the physical absolute offset in our symbol table
            result.offsets[tensor_id] = aligned_offset;

            // 6. Append the actual tensor bytes
            result.rodata_blob.insert(
                result.rodata_blob.end(), 
                byte_span.begin(), 
                byte_span.end()
            );

            ++packed_tensor_count_;
        }
    });

    if (!pass_result) {
        return std::unexpected(pass_result.error());
    }

    total_bytes_packed_ = result.rodata_blob.size();

    // Shrink the vector capacity to perfectly match the final size, returning unused RAM
    result.rodata_blob.shrink_to_fit();

    utility::Logger::Info(std::format(
        "WeightPacker: Successfully packed {} unique tensor(s). Total .rodata size: {} bytes", 
        packed_tensor_count_, total_bytes_packed_
    ));

    return result; // Relies on NRVO (Named Return Value Optimization) to prevent copies
}

}  // namespace seecpp::backend
