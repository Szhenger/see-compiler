#pragma once

#include "middle-end/pass_manager.hpp"
#include "middle-end/sir.hpp"

namespace seecpp::middle {

/**
 * ConvLoweringPass: High-Performance im2col + GEMM lowering.
 * 
 * DESIGN PRINCIPLES:
 * 1. Metadata-Driven Reshaping: Filter flattening is treated as a 0-cost 
 *    view transformation to avoid Arena allocation for "reshaped" weights.
 * 2. Batch-Aware MatMul: Lowers to BatchMatMul (BMM) to allow the Backend 
 *    to exploit hardware-specific parallelization across the N dimension.
 * 3. Iterator-Safe Transformation: Uses a "Replace-and-Advance" strategy 
 *    to maintain SIR block integrity during mutation.
 */

class ConvLoweringPass final : public IPass {
public:
    ConvLoweringPass() = default;

    std::string_view name() const override { return "sc_high.conv_to_low_matmul"; }

    bool requiresValidation() const override { return true; }

    [[nodiscard]]
    PassResult runOnBlock(sir::Block& block) override;

private:

    [[nodiscard]]
    PassResult lowerConv(sir::Block& block, sir::Operation* conv_op);
};

} // namespace seecpp::middle
