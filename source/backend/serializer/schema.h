#ifndef SEECPP_BACKEND_SRC_SERIALIZATION_SCHEMA_H_
#define SEECPP_BACKEND_SRC_SERIALIZATION_SCHEMA_H_

#include <cstdint>

namespace seecpp::backend {

// Magic bytes: "SEEC" (Little-endian layout)
constexpr uint32_t kSeeMagic = 0x43454553; 
constexpr uint32_t kCurrentVersion = 1;

/// @brief The bare-metal header at the very beginning of the .see file.
struct alignas(8) FileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t arena_size;      // Total dynamic memory the runtime must allocate
    uint64_t text_offset;     // Absolute byte offset to the instruction array
    uint64_t text_size;       // Number of instructions in the text section
    uint64_t rodata_offset;   // Absolute byte offset to the packed weights
    uint64_t rodata_size;     // Size of the packed weights in bytes
};

/// @brief A fixed-size instruction struct for O(1) runtime dispatch.
/// Limiting inputs/outputs to a fixed array avoids pointer chasing at runtime.
struct alignas(8) SerializedInstruction {
    uint16_t opcode;          // Maps directly to BackendOpcode
    uint16_t num_inputs;
    uint16_t num_outputs;
    uint16_t reserved;        // Padding for 8-byte alignment
    
    // Up to 4 inputs and 2 outputs per op (sufficient for GEMM, Conv, etc.)
    int64_t inputs[4];        
    int64_t outputs[2];
};

}  // namespace seecpp::backend

#endif  // SEECPP_BACKEND_SRC_SERIALIZATION_SCHEMA_H_
