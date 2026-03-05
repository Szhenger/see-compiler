#pragma once

#include "backend/codegen.hpp"
#include "middle-end/sir.hpp"
#include "utility/weight_buffer.hpp"

#include <expected>
#include <string_view>

// =============================================================================
// WeightFoldingPass
//
// Performs the deferred arithmetic for Conv+BN fusion.
//
// Context:
//   OperatorFusionPass stamps four attributes on each fused Conv op and
//   removes the BN op, but does NOT touch the weight tensors. The actual
//   folding is deferred because it requires WeightBuffer access, which
//   middle-end passes do not hold.
//
// This pass reads those attributes and computes:
//
//   gamma  = scale[f]                       (BN scale, per output channel)
//   beta   = bias[f]                        (BN bias, per output channel)
//   mu     = mean[f]                        (BN running mean)
//   sigma2 = var[f]                         (BN running variance)
//   eps    = bn_epsilon attribute
//
//   std_inv[f] = 1 / sqrt(sigma2[f] + eps)  (precomputed per channel)
//
//   For each output channel f:
//     W_fused[f, :, :, :] = W[f, :, :, :] * (gamma[f] * std_inv[f])
//     b_fused[f]           = (b[f] - mu[f]) * gamma[f] * std_inv[f] + beta[f]
//
// If the Conv had no bias, b_fused is initialised from zero.
//
// Folded tensors are written back to WeightBuffer under new keys:
//   "<original_weight_id>__bn_folded"
//   "<original_bias_id>__bn_folded"   (created if no original bias)
//
// The Conv op's filter operand id attribute is updated to the new key
// so that the codegen target reads the folded tensor automatically.
//
// Precondition: OperatorFusionPass has run (fused_bn=1 attribute present).
// Postcondition: no sc_high.conv2d op retains a live fused_bn attribute
//                without a corresponding folded tensor in WeightBuffer.
// =============================================================================

namespace seecpp::backend {

class WeightFoldingPass {
public:
    WeightFoldingPass() = default;

    std::string_view name() const { return "WeightFoldingPass"; }

    /// Run weight folding over `block`.
    /// Reads from and writes to `weights`.
    [[nodiscard]]
    std::expected<void, CodegenError>
    run(sir::Block& block, utility::WeightBuffer& weights);

private:
    /// Fold BN parameters into the filter and bias of one Conv op.
    [[nodiscard]]
    std::expected<void, CodegenError>
    foldConv(sir::Operation* conv_op, utility::WeightBuffer& weights);

    int folded_count_ = 0;
};

} // namespace seecpp::backend