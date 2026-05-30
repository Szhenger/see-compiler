#include "include/backend/codegen_driver.h"

// The pipeline components we built
#include "src/lowering/instruction_selector.h"
#include "src/memory/offset_binder.h"
#include "src/weights/weight_packer.h"
#include "src/serialization/serializer.h"

// Utilities
#include "include/utility/logger.h"
#include "seecpp/utility/weight_buffer.h"
#include "seecpp/sir/sir.h"

#include <format>

namespace seecpp::backend {

std::expected<void, CodegenError> CodegenDriver::Run(
    sir::Block& block,
    const utility::WeightBuffer& weights,
    std::string_view output_file) 
{
    utility::Logger::Info(std::format(
        "CodegenDriver: Starting AOT compilation to '{}'", output_file
    ));

    // =========================================================================
    // Phase 1: Instruction Selection
    // Lowers abstract math operations into hardware-specific execution opcodes.
    // =========================================================================
    utility::Logger::Info("CodegenDriver: [1/4] Running Instruction Selector...");
    InstructionSelector selector;
    if (auto res = selector.Run(block); !res) {
        return std::unexpected(CodegenError{
            "instruction_selection", 
            std::format("Failed to select instructions: {}", res.error().message)
        });
    }

    // =========================================================================
    // Phase 2: Memory Arena Binding
    // Calculates absolute byte offsets for all intermediate tensors.
    // =========================================================================
    utility::Logger::Info("CodegenDriver: [2/4] Running Offset Binder...");
    OffsetBinder binder;
    auto bind_result = binder.Run(block);
    if (!bind_result) {
        return std::unexpected(CodegenError{
            "offset_binding", 
            std::format("Failed to bind memory offsets: {}", bind_result.error().message)
        });
    }
    const uint64_t required_arena_size = bind_result.value();

    // =========================================================================
    // Phase 3: Weight Packing
    // Flattens, deduplicates, and aligns constants into a binary blob.
    // =========================================================================
    utility::Logger::Info("CodegenDriver: [3/4] Running Weight Packer...");
    WeightPacker packer;
    auto pack_result = packer.Run(block, weights);
    if (!pack_result) {
        return std::unexpected(CodegenError{
            "weight_packing", 
            std::format("Failed to pack weights: {}", pack_result.error().message)
        });
    }
    // Extract the packed payload and symbol table
    const PackedWeights packed_data = std::move(pack_result.value());

    // =========================================================================
    // Phase 4: Serialization
    // Writes the headers, instruction array, and weights to disk.
    // =========================================================================
    utility::Logger::Info("CodegenDriver: [4/4] Running Serializer...");
    Serializer serializer;
    if (auto res = serializer.Run(output_file, block, packed_data, required_arena_size); !res) {
        return std::unexpected(CodegenError{
            "serialization", 
            std::format("Failed to write binary file: {}", res.error().message)
        });
    }

    utility::Logger::Info("CodegenDriver: Compilation finished successfully.");
    return {};
}

}  // namespace seecpp::backend
