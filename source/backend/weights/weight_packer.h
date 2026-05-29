#ifndef SEECPP_BACKEND_SRC_WEIGHTS_WEIGHT_PACKER_H_
#define SEECPP_BACKEND_SRC_WEIGHTS_WEIGHT_PACKER_H_

#include <cstdint>
#include <expected>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
namespace seecpp::sir {
class Block;
class Operation;
}  // namespace seecpp::sir

namespace seecpp::utility {
class WeightBuffer;
}

namespace seecpp::backend {

struct CodegenError;

/// @brief Represents the compiled read-only data section and its symbol table.
struct PackedWeights {
    /// @brief The flattened, aligned, raw byte array of all constant weights.
    std::vector<uint8_t> rodata_blob;
    /// @brief Maps a tensor ID to its exact byte offset within the rodata_blob.
    std::unordered_map<std::string, uint64_t> offsets;
};

/// @brief Flattens and aligns mathematical tensors into a bare-metal binary blob.
class WeightPacker {
 public:
    // 64-byte alignment is mandatory to avoid AVX-512 unaligned load penalties.
    static constexpr size_t kDefaultAlignment = 64;

    WeightPacker() = default;

    /// @brief Scans the IR block, extracts referenced weights, and packs them.
    /// @param block The Middle-End optimized IR block.
    /// @param weights The buffer containing the raw parsed constants.
    /// @param alignment The byte boundary to align each tensor to.
    /// @return The compiled PackedWeights struct, or a CodegenError.
    [[nodiscard]] std::expected<PackedWeights, CodegenError> Run(
        sir::Block& block, 
        const utility::WeightBuffer& weights,
        size_t alignment = kDefaultAlignment);

 private:
    size_t packed_tensor_count_ = 0;
    size_t total_bytes_packed_ = 0;
};

}  // namespace seecpp::backend

#endif  // SEECPP_BACKEND_SRC_WEIGHTS_WEIGHT_PACKER_H_
