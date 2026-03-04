#include "include/middle-end/conv_lowering_pass.hpp"
#include "include/utility/logger.hpp"

#include <cassert>

namespace seecpp::middle {

std::expected<void, PassError>
ConvLoweringPass::runOnBlock(sir::Block& block) {

    // Collect all sc_high.conv2d ops first — we cannot modify the op list
    // while iterating it. This is the standard collect-then-rewrite pattern
    // used by MLIR's RewritePatternSet and TVM's sequential passes.
    std::vector<sir::Operation*> to_lower;
    block.walk([&](sir::Operation* op) {
        if (op->mnemonic() == "sc_high.conv2d")
            to_lower.push_back(op);
    });

    utility::Logger::info(
        "ConvLoweringPass: lowering " +
        std::to_string(to_lower.size()) + " conv2d op(s)");

    for (sir::Operation* conv_op : to_lower) {
        if (auto result = lowerConv(block, conv_op); !result)
            return result;
    }

    return {};
}

std::expected<void, PassError>
ConvLoweringPass::lowerConv(sir::Block& block, sir::Operation* conv_op) {

    // --- 1. Extract operands and attributes ---
    if (conv_op->numOperands() < 2)
        return std::unexpected(PassError{
            std::string(name()),
            "Conv2D op has fewer than 2 operands"});

    sir::Value* input  = conv_op->operand(0);  // [N, C, H, W]
    sir::Value* filter = conv_op->operand(1);  // [F, C, KH, KW]
    sir::Value* bias   = conv_op->numOperands() > 2
                             ? conv_op->operand(2) : nullptr;

    auto strides   = conv_op->getAttrAs<std::vector<int64_t>>("strides")
                         .value_or(std::vector<int64_t>{1, 1});
    auto pads      = conv_op->getAttrAs<std::vector<int64_t>>("pads")
                         .value_or(std::vector<int64_t>{0, 0, 0, 0});
    auto dilations = conv_op->getAttrAs<std::vector<int64_t>>("dilations")
                         .value_or(std::vector<int64_t>{1, 1});

    const auto& in_dims  = input->shape().dims;
    const auto& fil_dims = filter->shape().dims;

    if (in_dims.size() != 4 || fil_dims.size() != 4)
        return std::unexpected(PassError{
            std::string(name()),
            "ConvLowering: expected rank-4 input and filter (NCHW)"});

    const int64_t N  = in_dims[0];
    const int64_t C  = in_dims[1];
    const int64_t KH = fil_dims[2];
    const int64_t KW = fil_dims[3];
    const int64_t F  = fil_dims[0];

    // Reuse the shape formula from ShapeInferencePass.
    auto conv_dim = [](int64_t in, int64_t k, int64_t s,
                       int64_t pb, int64_t pe, int64_t d) -> int64_t {
        if (in == sir::Shape::kDynamic) return sir::Shape::kDynamic;
        return (in + pb + pe - d * (k - 1) - 1) / s + 1;
    };

    const int64_t out_H = conv_dim(in_dims[2], KH, strides[0],
                                    pads[0], pads[2], dilations[0]);
    const int64_t out_W = conv_dim(in_dims[3], KW, strides[1],
                                    pads[1], pads[3], dilations[1]);

    // --- 2. Emit sc_low.im2col ---
    // Output shape: [N, C*KH*KW, out_H*out_W]
    const int64_t col_rows = C * KH * KW;
    const int64_t col_cols = (out_H == sir::Shape::kDynamic ||
                               out_W == sir::Shape::kDynamic)
                                 ? sir::Shape::kDynamic
                                 : out_H * out_W;

    auto im2col_op = block.appendOp("sc_low.im2col");
    im2col_op->addOperand(input);
    im2col_op->setAttribute("kernel_shape",
                             std::vector<int64_t>{KH, KW});
    im2col_op->setAttribute("strides",   strides);
    im2col_op->setAttribute("pads",      pads);
    im2col_op->setAttribute("dilations", dilations);
    sir::Value* col_matrix = im2col_op->addResult(
        "", input->dtype(), sir::Shape{{N, col_rows, col_cols}});

    // --- 3. Emit sc_low.reshape (flatten filter to [F, C*KH*KW]) ---
    auto reshape_op = block.appendOp("sc_low.reshape");
    reshape_op->addOperand(filter);
    reshape_op->setAttribute("target_shape",
                              std::vector<int64_t>{F, col_rows});
    sir::Value* filter_matrix = reshape_op->addResult(
        "", filter->dtype(), sir::Shape{{F, col_rows}});

    // --- 4. Emit sc_low.matmul (filter_matrix @ col_matrix) ---
    // Per-batch: [F, C*KH*KW] @ [C*KH*KW, out_H*out_W] -> [F, out_H*out_W]
    // Batched:   [N, F, out_H*out_W]
    auto matmul_op = block.appendOp("sc_low.matmul");
    matmul_op->addOperand(filter_matrix);
    matmul_op->addOperand(col_matrix);

    sir::Value* matmul_out = matmul_op->addResult(
        "", input->dtype(), sir::Shape{{N, F, col_cols}});

    // --- 5. Optionally emit sc_low.add for bias ---
    sir::Value* lowered_out = matmul_out;
    if (bias) {
        auto bias_op = block.appendOp("sc_low.add");
        bias_op->addOperand(matmul_out);
        bias_op->addOperand(bias);
        lowered_out = bias_op->addResult(
            "", matmul_out->dtype(), matmul_out->shape());
    }

    // --- 6. Wire conv result users to the new lowered output ---
    // replaceAllUsesWith updates every downstream op that consumed the
    // conv result to now consume the im2col+matmul result instead.
    if (conv_op->numResults() > 0)
        conv_op->result(0)->replaceAllUsesWith(lowered_out);

    // --- 7. Erase the original sc_high.conv2d ---
    block.removeOp(conv_op);

    utility::Logger::info(
        "ConvLoweringPass: conv2d -> im2col + reshape + matmul" +
        std::string(bias ? " + add(bias)" : ""));

    return {};
}

} // namespace seecpp::middle