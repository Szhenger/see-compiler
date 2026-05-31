#ifndef SEECPP_BACKEND_SCHEMA_H_
#define SEECPP_BACKEND_SCHEMA_H_

#include <cstdint>

namespace seecpp::backend {

// =============================================================================
// Magic Constants & Versioning
// =============================================================================

// "SEE!" in little-endian ASCII (0x21454553)
inline constexpr uint32_t kSeeMagic = 0x21454553; 

// Increment this whenever the schema structs change to prevent segfaults
inline constexpr uint32_t kCurrentVersion = 1;

// =============================================================================
// Memory Layout Definitions
// =============================================================================

// Ensure strict packing so the C++ compiler doesn't insert hidden padding bytes.
#pragma pack(push, 1)

/// @brief The 64-byte master header at byte 0 of the .see file.
struct FileHeader {
    uint32_t magic;          // 4 bytes: kSeeMagic
    uint32_t version;        // 4 bytes: kCurrentVersion
    
    uint64_t arena_size;     // 8 bytes: Total bytes to aligned_alloc() at runtime
    
    uint64_t text_offset;    // 8 bytes: Absolute file offset to SerializedInstruction array
    uint64_t text_size;      // 8 bytes: Number of instructions to execute
    
    uint64_t rodata_offset;  // 8 bytes: Absolute file offset to the packed weights
    uint64_t rodata_size;    // 8 bytes: Size of the weight blob in bytes
    
    uint64_t reserved[2];    // 16 bytes: Padding to pad the header to exactly 64 bytes
};

/// @brief A 64-byte instruction block, explicitly designed to fit in a single L1 cache line.
struct SerializedInstruction {
    uint16_t opcode;         // 2 bytes: Hardware operation (e.g., kGemv, kRelu)
    uint16_t flags;          // 2 bytes: Execution flags (e.g., in-place mutation allowed)
    uint32_t padding;        // 4 bytes: Padding to keep 64-bit alignment for the arrays
    
    // Arrays of 64-bit integers. 
    // These can hold absolute byte offsets into the arena/rodata, or be bit-packed 
    // with metadata (like packing tensor dimensions `M` and `N` into a single uint64_t).
    uint64_t inputs[4];      // 32 bytes: Input operand offsets or static metadata
    uint64_t outputs[3];     // 24 bytes: Destination arena offsets
};

#pragma pack(pop)

// =============================================================================
// Compile-Time Layout Validation
// =============================================================================
// If any of these fail, the compiler will refuse to build, ensuring we never 
// accidentally break the ABI (Application Binary Interface).

static_assert(sizeof(FileHeader) == 64, 
    "FileHeader must be exactly 64 bytes to guarantee cache-line alignment.");
    
static_assert(sizeof(SerializedInstruction) == 64, 
    "SerializedInstruction must be exactly 64 bytes to prevent cache-line spanning.");

}  // namespace seecpp::backend

#endif  // SEECPP_BACKEND_SCHEMA_H_
