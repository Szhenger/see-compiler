#include "source/serialization/serializer.h"
#include "source/serialization/schema.h"
#include "source/weights/weight_packer.h" // For PackedWeights struct
#include "include/utility/logger.h"

#include "seecpp/sir/sir.h"

#include <format>
#include <fstream>
#include <vector>

namespace seecpp::backend {

namespace {
// Helper to pad file streams to specific byte boundaries for cache-line alignment
void WritePadding(std::ofstream& file, size_t current_offset, size_t alignment) {
    const size_t aligned_offset = (current_offset + alignment - 1) & ~(alignment - 1);
    const size_t padding_bytes = aligned_offset - current_offset;
    if (padding_bytes > 0) {
        std::vector<char> pad(padding_bytes, 0);
        file.write(pad.data(), pad.size());
    }
}
}  // namespace

std::expected<void, CodegenError> Serializer::Run(
    std::string_view file_path, 
    const sir::Block& block, 
    const PackedWeights& weights,
    uint64_t required_arena_size) 
{
    utility::Logger::Info(std::format("Serializer: Writing binary to '{}'", file_path));

    // --- 1. Extract and Validate Instructions from IR ---
    std::vector<SerializedInstruction> text_section;
    std::expected<void, CodegenError> pass_result = {};

    block.walk([&](const sir::Operation* op) {
        if (!pass_result) return;

        // Fetch attributes bound by previous passes
        auto opcode_opt = op->GetAttribute<int64_t>("runtime_opcode");
        auto inputs_opt = op->GetAttribute<std::vector<int64_t>>("input_offsets");
        auto outputs_opt = op->GetAttribute<std::vector<int64_t>>("output_offsets");

        if (!opcode_opt || !inputs_opt || !outputs_opt) {
            pass_result = std::unexpected(CodegenError{
                "serialization", 
                std::format("Operation '{}' is missing required lowering attributes. "
                            "Did the Selector and OffsetBinder run?", op->mnemonic())
            });
            return;
        }

        const auto& inputs = inputs_opt.value();
        const auto& outputs = outputs_opt.value();

        if (inputs.size() > 4 || outputs.size() > 2) {
            pass_result = std::unexpected(CodegenError{
                "serialization", 
                std::format("Operation '{}' exceeds hardware struct limit (max 4 inputs, 2 outputs).", 
                            op->mnemonic())
            });
            return;
        }

        // Populate the physical struct
        SerializedInstruction inst{};
        inst.opcode = static_cast<uint16_t>(opcode_opt.value());
        inst.num_inputs = static_cast<uint16_t>(inputs.size());
        inst.num_outputs = static_cast<uint16_t>(outputs.size());
        inst.reserved = 0;

        for (size_t i = 0; i < inputs.size(); ++i) inst.inputs[i] = inputs[i];
        for (size_t i = 0; i < outputs.size(); ++i) inst.outputs[i] = outputs[i];

        text_section.push_back(inst);
    });

    if (!pass_result) return pass_result;

    // --- 2. Calculate Layout Offsets ---
    FileHeader header{};
    header.magic = kSeeMagic;
    header.version = kCurrentVersion;
    header.arena_size = required_arena_size;
    header.text_size = text_section.size();
    header.rodata_size = weights.rodata_blob.size();

    // Text section follows immediately after the header
    header.text_offset = sizeof(FileHeader);
    
    // Rodata section must be 64-byte aligned for AVX-512 loading
    size_t end_of_text = header.text_offset + (header.text_size * sizeof(SerializedInstruction));
    header.rodata_offset = (end_of_text + 63) & ~63; 

    // --- 3. Write to Disk ---
    std::ofstream out(std::string(file_path), std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        return std::unexpected(CodegenError{
            "io_error", 
            std::format("Failed to open output file '{}' for writing.", file_path)
        });
    }

    // Write Header
    out.write(reinterpret_cast<const char*>(&header), sizeof(FileHeader));

    // Write Text Section (Instructions)
    out.write(reinterpret_cast<const char*>(text_section.data()), 
              text_section.size() * sizeof(SerializedInstruction));

    // Pad to 64-byte boundary
    WritePadding(out, static_cast<size_t>(out.tellp()), 64);

    // Write Rodata Section (Weights)
    if (!weights.rodata_blob.empty()) {
        out.write(reinterpret_cast<const char*>(weights.rodata_blob.data()), 
                  weights.rodata_blob.size());
    }

    if (out.fail()) {
        return std::unexpected(CodegenError{
            "io_error", 
            "A filesystem error occurred while writing the binary payload."
        });
    }

    utility::Logger::Info(std::format(
        "Serializer: Build complete. Output size: {} bytes. (Instructions: {}, Arena: {} bytes)",
        static_cast<size_t>(out.tellp()), text_section.size(), required_arena_size
    ));

    return {};
}

}  // namespace seecpp::backend
