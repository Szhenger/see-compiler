#ifndef SEECPP_BACKEND_SRC_SERIALIZATION_SERIALIZER_H_
#define SEECPP_BACKEND_SRC_SERIALIZATION_SERIALIZER_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

// Forward declarations
namespace seecpp::sir {
class Block;
}

namespace seecpp::backend {

struct CodegenError;
struct PackedWeights;

/// @brief Writes the lowered, bound, and packed IR out to a physical binary file.
class Serializer {
 public:
    Serializer() = default;

    /// @brief Compiles the final state into a .see file.
    /// @param file_path Destination path on disk.
    /// @param block The fully lowered SIR block containing opcodes and offsets.
    /// @param weights The compiled read-only data section from the WeightPacker.
    /// @param required_arena_size Total dynamic memory needed for intermediate tensors.
    /// @return Expected void on success, or a CodegenError on failure.
    [[nodiscard]] std::expected<void, CodegenError> Run(
        std::string_view file_path, 
        const sir::Block& block, 
        const PackedWeights& weights,
        uint64_t required_arena_size);
};

}  // namespace seecpp::backend

#endif  // SEECPP_BACKEND_SRC_SERIALIZATION_SERIALIZER_H_
