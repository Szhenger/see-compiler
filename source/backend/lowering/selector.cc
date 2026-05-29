#include "src/lowering/selector.h"
#include "include/utility/logger.hpp"

// Assuming your framework provides the core IR definitions via this header
#include "seecpp/sir/sir.h" 

#include <format>
#include <string_view>

namespace seecpp::backend {

std::expected<void, CodegenError>
InstructionSelector::run(sir::Block& block, TargetArch arch) {
    lowered_count_ = 0;

    std::string_view arch_str = (arch == TargetArch::x86_64_AVX512) ? "AVX-512" :
                                (arch == TargetArch::ARM_NEON)      ? "NEON" : 
                                                                      "Generic Scalar";

    utility::Logger::info(std::format(
        "InstructionSelector: Starting lowering pass for target architecture '{}'", 
        arch_str
    ));

    // Capture errors from within the lambda during the IR traversal walk
    std::expected<void, CodegenError> pass_result = {};

    block.walk([&](sir::Operation* op) {
        // Short-circuit if an error was previously encountered
        if (!pass_result) return;

        if (auto res = lowerOperation(op, arch); !res) {
            pass_result = res;
        }
    });

    // Bubble up any errors caught during the graph traversal
    if (!pass_result) {
        return pass_result;
    }

    utility::Logger::info(std::format(
        "InstructionSelector: successfully mapped {} operation(s) to target opcodes", 
        lowered_count_
    ));

    return {};
}

std::expected<void, CodegenError>
InstructionSelector::lowerOperation(sir::Operation* op, TargetArch arch) {
    const std::string mnemonic = op->mnemonic();
    BackendOpcode selected_opcode = BackendOpcode::INVALID;

    // --- 1. Lower Matrix Multiplication (GEMM) ---
    if (mnemonic == "sc_low.matmul") {
        if (arch == TargetArch::x86_64_AVX512) {
            selected_opcode = BackendOpcode::AVX512_GEMM_FP32;
        } else if (arch == TargetArch::ARM_NEON) {
            selected_opcode = BackendOpcode::NEON_GEMM_FP32;
        } else {
            selected_opcode = BackendOpcode::SCALAR_GEMM_FP32;
        }
    }
    // --- 2. Lower Convolutions ---
    else if (mnemonic == "sc_low.conv2d" || mnemonic == "sc_high.conv2d") {
        if (arch == TargetArch::x86_64_AVX512) {
            selected_opcode = BackendOpcode::AVX512_CONV2D_FP32;
        } else if (arch == TargetArch::ARM_NEON) {
            selected_opcode = BackendOpcode::NEON_CONV2D_FP32;
        } else {
            selected_opcode = BackendOpcode::SCALAR_CONV2D_FP32;
        }
    }
    // --- 3. Lower Activations ---
    else if (mnemonic == "sc_low.relu") {
        if (arch == TargetArch::x86_64_AVX512) {
            selected_opcode = BackendOpcode::AVX512_RELU_FP32;
        } else if (arch == TargetArch::ARM_NEON) {
            selected_opcode = BackendOpcode::NEON_RELU_FP32;
        } else {
            selected_opcode = BackendOpcode::SCALAR_RELU_FP32;
        }
    }
    // --- 4. Unrecognized Operation Error ---
    else {
        return std::unexpected(CodegenError{
            "instruction_selection",
            std::format("Backend selector encountered unhandled instruction mnemonic '{}'", mnemonic)
        });
    }

    // --- 5. Bind the raw numeric value back to the IR operation attributes ---
    // Storing as a standard primitive type allows the final serializer to read 
    // it without needing complex deserialization dependencies.
    op->setAttribute("runtime_opcode", static_cast<int64_t>(selected_opcode));

    ++lowered_count_;
    return {};
}

} // namespace seecpp::backend
