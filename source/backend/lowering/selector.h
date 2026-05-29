#ifndef SEECPP_BACKEND_SRC_LOWERING_SELECTOR_H_
#define SEECPP_BACKEND_SRC_LOWERING_SELECTOR_H_

#include <expected>
#include <string>
#include <cstdint>

// Forward declarations for the SeeC++ core IR
namespace seecpp::sir {
    class Block;
    class Operation;
}

namespace seecpp::backend {

// Forward declare or reference your existing CodegenError from weight_folding_pass
struct CodegenError;

/// @brief Defines the target hardware architecture for kernel selection.
enum class TargetArch {
    x86_64_AVX512,
    ARM_NEON,
    Generic_Scalar
};

/// @brief Internal enumeration of backend opcodes to avoid external runtime dependencies.
enum class BackendOpcode : uint16_t {
    INVALID            = 0,
    
    SCALAR_GEMM_FP32   = 1,
    AVX512_GEMM_FP32   = 2,
    NEON_GEMM_FP32     = 3,

    SCALAR_CONV2D_FP32 = 4,
    AVX512_CONV2D_FP32 = 5,
    NEON_CONV2D_FP32   = 6,

    SCALAR_RELU_FP32   = 7,
    AVX512_RELU_FP32   = 8,
    NEON_RELU_FP32     = 9
};

/// @brief Analyzes SIR nodes and maps them to target-specific backend opcodes.
class InstructionSelector {
public:
    InstructionSelector() = default;

    /// @brief Lowers a generic SIR block into an opcode-mapped execution plan.
    /// @param block The Middle-End optimized IR block.
    /// @param arch The target hardware architecture.
    /// @return Expected void on success, or a CodegenError on failure.
    std::expected<void, CodegenError> run(sir::Block& block, TargetArch arch);

private:
    /// @brief Resolves a single SIR operation into an internal hardware opcode.
    std::expected<void, CodegenError> lowerOperation(sir::Operation* op, TargetArch arch);

    size_t lowered_count_ = 0;
};

} // namespace seecpp::backend

#endif // SEECPP_BACKEND_SRC_LOWERING_SELECTOR_H_
