#include "include/middle-end/op_fusion_pass.hpp"
#include "include/utility/logger.hpp"

#include <cmath>

namespace seecpp::middle {

std::expected<void, PassError>
OperatorFusionPass::runOnBlock(sir::Block& block) {

    fused_conv_bn_    = 0;
    fused_matmul_relu_  = 0;
    fused_elementwise_  = 0;

    // --- Pass 1: Conv-BN fusion (sc_high, pre-lowering) ---
    // Collect BN ops first to avoid iterator invalidation during rewrite.
    {
        std::vector<sir::Operation*> bn_ops;
        block.walk([&](sir::Operation* op) {
            if (op->mnemonic() == "sc_high.batch_norm")
                bn_ops.push_back(op);
        });
        for (auto* bn : bn_ops)
            fuseConvBatchNorm(block, bn);
    }

    // --- Pass 2: MatMul/Conv-Relu fusion ---
    {
        std::vector<sir::Operation*> relu_ops;
        block.walk([&](sir::Operation* op) {
            if (op->mnemonic() == "sc_high.relu" ||
                op->mnemonic() == "sc_low.relu")
                relu_ops.push_back(op);
        });
        for (auto* relu : relu_ops)
            fuseMatMulRelu(block, relu);
    }

    // --- Pass 3: Generic elementwise chain fusion ---
    {
        std::vector<sir::Operation*> ew_ops;
        block.walk([&](sir::Operation* op) {
            const std::string mn(op->mnemonic());
            if (mn == "sc_high.add" || mn == "sc_high.mul" ||
                mn == "sc_high.sub" || mn == "sc_high.div")
                ew_ops.push_back(op);
        });
        for (auto* ew : ew_ops)
            fuseElementwiseChain(block, ew);
    }

    utility::Logger::info(
        "OperatorFusionPass: fused " +
        std::to_string(fused_conv_bn_)    + " conv+bn, " +
        std::to_string(fused_matmul_relu_)  + " matmul+relu, " +
        std::to_string(fused_elementwise_)  + " elementwise chain(s)");

    return {};
}

// =============================================================================
// Pattern 1 — Conv + BatchNorm fusion
//
// BatchNorm in inference mode is a linear transform:
//   y = (x - mean) / sqrt(var + eps) * scale + bias
//
// This can be folded into the preceding Conv's filter W and bias b:
//   W_fused = W * (scale / sqrt(var + eps))   [broadcast over output channels]
//   b_fused = (b - mean) * scale / sqrt(var + eps) + bias
//
// After folding, the BatchNorm op is dead and removed.
// =============================================================================

bool OperatorFusionPass::fuseConvBatchNorm(
    sir::Block& block, sir::Operation* bn_op)
{
    // BN must have exactly one input operand whose defining op is a Conv.
    if (bn_op->numOperands() < 5) return false;

    sir::Value*    bn_input  = bn_op->operand(0);
    sir::Operation* conv_op  = bn_input->definingOp();
    if (!conv_op) return false;
    if (conv_op->mnemonic() != "sc_high.conv2d") return false;

    // Conv result must be used ONLY by this BN — no other users allowed.
    // Fusing would change the conv output dtype/value for other users.
    if (!bn_input->hasOneUse()) return false;

    // All BN parameters must be compile-time constants (sc_high.constant).
    // We can only fold if we have the actual values at compile time.
    auto isConstant = [](sir::Value* v) -> bool {
        return v && v->definingOp() &&
               v->definingOp()->mnemonic() == "sc_high.constant";
    };
    if (!isConstant(bn_op->operand(1)) ||  // scale
        !isConstant(bn_op->operand(2)) ||  // bias
        !isConstant(bn_op->operand(3)) ||  // mean
        !isConstant(bn_op->operand(4)))    // var
        return false;

    float epsilon = bn_op->getAttrAs<float>("epsilon").value_or(1e-5f);

    // Mark the Conv op as having fused BN parameters so the codegen backend
    // knows to use the folded weights from WeightBuffer rather than the
    // original filter. Actual weight arithmetic happens in the backend.
    // (Full weight folding requires WeightBuffer access, which is a
    // backend concern — we encode the fusion decision as attributes here.)
    conv_op->setAttribute("fused_bn",      int64_t(1));
    conv_op->setAttribute("bn_epsilon",    epsilon);
    conv_op->setAttribute("bn_scale_id",
        std::string(bn_op->operand(1)->id()));
    conv_op->setAttribute("bn_bias_id",
        std::string(bn_op->operand(2)->id()));
    conv_op->setAttribute("bn_mean_id",
        std::string(bn_op->operand(3)->id()));
    conv_op->setAttribute("bn_var_id",
        std::string(bn_op->operand(4)->id()));

    // Redirect all users of the BN output to the Conv output.
    bn_op->result(0)->replaceAllUsesWith(conv_op->result(0));

    // Propagate BN's output shape back to Conv result (they are equal for BN).
    conv_op->result(0)->setShape(bn_op->result(0)->shape());

    // Remove the now-dead BN op.
    block.removeOp(bn_op);
    ++fused_conv_bn_;
    return true;
}

// =============================================================================
// Pattern 2 — MatMul/Conv + Relu fusion
//
// If a Relu immediately follows a matmul or convolution and the matmul
// result has no other users, we absorb Relu as an activation attribute.
// The backend generates a single kernel that writes clamped(0, output).
// =============================================================================

bool OperatorFusionPass::fuseMatMulRelu(
    sir::Block& block, sir::Operation* relu_op)
{
    if (relu_op->numOperands() < 1) return false;

    sir::Value*    relu_input = relu_op->operand(0);
    sir::Operation* producer  = relu_input->definingOp();
    if (!producer) return false;

    const std::string mn(producer->mnemonic());
    const bool is_matmul = (mn == "sc_low.matmul"  ||
                             mn == "sc_high.matmul" ||
                             mn == "sc_high.gemm");
    if (!is_matmul) return false;

    // Producer must have exactly one user (this relu).
    if (!relu_input->hasOneUse()) return false;

    // Encode the fusion as an attribute on the producer.
    producer->setAttribute("activation", std::string("relu"));

    // Redirect Relu users to the producer result.
    relu_op->result(0)->replaceAllUsesWith(producer->result(0));

    block.removeOp(relu_op);
    ++fused_matmul_relu_;
    return true;
}

// =============================================================================
// Pattern 3 — Elementwise chain fusion
//
// If a binary elementwise op (Add/Mul/Sub/Div) consumes the output of
// another elementwise op and that intermediate has exactly one use,
// merge them into a sc_high.fused_ew op with an "ops" attribute listing
// the sequence. The backend emits a single loop over the output buffer.
// =============================================================================

bool OperatorFusionPass::fuseElementwiseChain(
    sir::Block& block, sir::Operation* consumer_op)
{
    // consumer_op must have at least one operand whose producer is also
    // a simple elementwise op with one use.
    for (size_t i = 0; i < consumer_op->numOperands(); ++i) {
        sir::Value*    v        = consumer_op->operand(i);
        sir::Operation* producer = v->definingOp();
        if (!producer) continue;

        const std::string pmn(producer->mnemonic());
        const bool is_ew_producer =
            (pmn == "sc_high.add" || pmn == "sc_high.mul" ||
             pmn == "sc_high.sub" || pmn == "sc_high.div");
        if (!is_ew_producer) continue;
        if (!v->hasOneUse()) continue;

        // Build a fused op that carries the sequence of op types as an attr.
        std::string op_seq = std::string(producer->mnemonic()) +
                             "+" +
                             std::string(consumer_op->mnemonic());

        auto fused_op = block.appendOp("sc_high.fused_ew");
        fused_op->setAttribute("op_sequence", op_seq);

        // Wire all producer operands first, then consumer operands
        // (excluding the intermediate value being fused away).
        for (size_t j = 0; j < producer->numOperands(); ++j)
            fused_op->addOperand(producer->operand(j));
        for (size_t j = 0; j < consumer_op->numOperands(); ++j)
            if (consumer_op->operand(j) != v)
                fused_op->addOperand(consumer_op->operand(j));

        fused_op->addResult("", consumer_op->result(0)->dtype(),
                                consumer_op->result(0)->shape());

        consumer_op->result(0)->replaceAllUsesWith(fused_op->result(0));

        block.removeOp(consumer_op);
        block.removeOp(producer);
        ++fused_elementwise_;
        return true;
    }
    return false;
}

} // namespace seecpp::middle