#pragma once

#include "include/middle-end/pass_manager.hpp"
#include "include/middle-end/sir.hpp"

// =============================================================================
// OperatorFusionPass
//
// Fuses adjacent elementwise ops into a single kernel to eliminate
// intermediate memory round-trips — the single biggest source of memory
// bandwidth waste in DL inference on both CPU and GPU.
//
// Patterns detected (in priority order):
//
//   Pattern 1 — Conv-BN fusion (sc_high dialect, pre-lowering):
//     sc_high.conv2d + sc_high.batch_norm -> sc_high.conv2d (folded weights)
//     BatchNorm parameters are folded into the Conv filter and bias at
//     compile time, eliminating the BN op entirely.
//     Condition: BN must be in inference mode (running stats available).
//
//   Pattern 2 — Conv-Relu fusion (sc_low dialect, post-lowering):
//     sc_low.matmul + sc_high.relu -> sc_low.matmul_relu
//     Relu is absorbed as an activation attribute on the matmul op,
//     allowing the kernel to apply it in the same pass over the output buffer.
//
//   Pattern 3 — Elementwise chain fusion (sc_high dialect):
//     sc_high.add + sc_high.relu -> sc_high.add_relu
//     Any chain of elementwise ops with a single shared user can be merged.
//
// Precondition: ShapeInferencePass must have run (all shapes resolved).
// This pass should run BEFORE ConvLoweringPass for Pattern 1, and AFTER for
// Pattern 2.
// =============================================================================

namespace seecpp::middle {

class OperatorFusionPass final : public IPass {
public:
    OperatorFusionPass() = default;

    std::string_view name() const override { return "OperatorFusionPass"; }

    // Fusion restructures the graph significantly — validate after.
    bool requiresValidation() const override { return true; }

    [[nodiscard]]
    std::expected<void, PassError> runOnBlock(sir::Block& block) override;

private:
    // --- Pattern matchers ---

    /// Conv + BN fusion: folds BN scale/bias/mean/var into Conv filter+bias.
    /// Returns true if the pattern was matched and rewritten.
    bool fuseConvBatchNorm(sir::Block& block, sir::Operation* bn_op);

    /// MatMul/Conv result + Relu fusion: adds "activation=relu" attribute.
    bool fuseMatMulRelu(sir::Block& block, sir::Operation* relu_op);

    /// Generic elementwise chain fusion: merges producer + consumer into
    /// a single fused op when the producer has exactly one user.
    bool fuseElementwiseChain(sir::Block& block, sir::Operation* consumer_op);

    // --- Statistics (surfaced in Logger output) ---
    int fused_conv_bn_   = 0;
    int fused_matmul_relu_ = 0;
    int fused_elementwise_ = 0;
};

} // namespace seecpp::middle