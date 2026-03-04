#pragma once

#include "middle-end/pass_manager.hpp"
#include "middle-end/sir.hpp"

// =============================================================================
// ConvLoweringPass
//
// Dialect transition: sc_high.conv2d -> sc_low.im2col + sc_low.matmul
//
// Algorithm (standard Im2Col lowering used by Caffe, cuDNN, NNPACK):
//
//   Given: sc_high.conv2d(input[N,C,H,W], filter[F,C,KH,KW])
//
//   Step 1 — Im2Col:
//     Unfold input patches into columns:
//       col_matrix : [N, C*KH*KW, out_H*out_W]
//
//   Step 2 — Filter reshape:
//     Flatten filter into a row matrix:
//       filter_matrix : [F, C*KH*KW]
//
//   Step 3 — MatMul:
//     filter_matrix @ col_matrix -> output[N, F, out_H, out_W]
//
// This lowering makes the convolution expressible as a single GEMM call,
// which backends (BLAS, cuBLAS, custom SIMD kernels) can optimise heavily.
// =============================================================================

namespace seecpp::middle {

class ConvLoweringPass final : public IPass {
public:
    ConvLoweringPass() = default;

    std::string_view name() const override { return "ConvLoweringPass"; }
    bool requiresValidation() const override { return true; }

    [[nodiscard]]
    std::expected<void, PassError> runOnBlock(sir::Block& block) override;

private:
    /// Lower a single sc_high.conv2d operation.
    /// Inserts sc_low.im2col + sc_low.reshape + sc_low.matmul into `block`
    /// immediately before `conv_op`, then removes `conv_op`.
    [[nodiscard]]
    std::expected<void, PassError> lowerConv(
        sir::Block&     block,
        sir::Operation* conv_op);
};

} // namespace seecpp::middle