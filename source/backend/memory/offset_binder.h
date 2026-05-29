#ifndef SEECPP_BACKEND_SRC_MEMORY_OFFSET_BINDER_H_
#define SEECPP_BACKEND_SRC_MEMORY_OFFSET_BINDER_H_

#include <expected>
#include <string>
#include <vector>
#include <cstdint>

// Forward declarations for the SeeC++ core IR
namespace seecpp::sir {
    class Block;
    class Operation;
}

// Forward declare the Middle-End's memory layout contract
namespace seecpp::middle_end {
    class ArenaLayout;
}

namespace seecpp::backend {

// Forward declare the established error structure
struct CodegenError;

/// @brief Translates abstract tensor IDs into absolute, hardcoded byte offsets.
class OffsetBinder {
public:
    OffsetBinder() = default;

    /// @brief Binds all operands and results in a block to absolute memory offsets.
    /// @param block The Middle-End optimized IR block.
    /// @param layout The finalized memory arena layout.
    /// @return Expected void on success, or a CodegenError on failure.
    [[nodiscard]] std::expected<void, CodegenError> run(
        sir::Block& block, 
        const middle_end::ArenaLayout& layout
    );

private:
    /// @brief Resolves input/output offsets for a single operation.
    [[nodiscard]] std::expected<void, CodegenError> bindOperation(
        sir::Operation* op, 
        const middle_end::ArenaLayout& layout
    );

    size_t bound_operations_ = 0;
};

} // namespace seecpp::backend

#endif // SEECPP_BACKEND_SRC_MEMORY_OFFSET_BINDER_H_
