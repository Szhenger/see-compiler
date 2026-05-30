#ifndef SEECPP_BACKEND_CODEGEN_DRIVER_H_
#define SEECPP_BACKEND_CODEGEN_DRIVER_H_

#include <expected>
#include <string_view>

// Forward declarations
namespace seecpp::sir {
class Block;
}

namespace seecpp::utility {
class WeightBuffer;
}

namespace seecpp::backend {

struct CodegenError {
    std::string phase;
    std::string message;
};

/// @brief Orchestrates the lowering, packing, and serialization of an ML model.
class CodegenDriver {
 public:
    CodegenDriver() = default;

    /// @brief Executes the complete backend compilation pipeline.
    /// @param block The optimized Middle-End IR graph.
    /// @param weights The raw constant weights.
    /// @param output_file The destination path for the .see executable.
    /// @return Expected void on success, or a CodegenError on failure.
    [[nodiscard]] std::expected<void, CodegenError> Run(
        sir::Block& block,
        const utility::WeightBuffer& weights,
        std::string_view output_file);
};

}  // namespace seecpp::backend

#endif  // SEECPP_BACKEND_CODEGEN_DRIVER_H_
