#include "middle-end/conv_lowering_pass.hpp"
#include "utility/logger.hpp"

#include <cassert>
#include <format>

namespace seecpp::middle {

PassResult ConvLoweringPass::runOnBlock(sir::Block& block) {
    // Collect all sc_high.conv2d ops first to safely iterate 
    // while we mutate the block structure.
    std::vector<sir::Operation*> to_lower;
    block.walk([&](sir::Operation* op) {
        if (op->mnemonic() == "sc_high.conv2d") {
            to_lower.push_back(op);
        }
    });

    if (to_lower.empty()) return {};

    utility::Logger::info(std::format("ConvLoweringPass: lowering {} conv2d op(s)", 
                                      to_lower.size()));

    for (sir::Operation* conv_op : to_lower) {
        if (auto result = lowerConv(block, conv_op); !result) {
            return result;
        }
    }

    return {};
}

PassResult ConvLoweringPass::lowerConv(sir::Block& block, sir::Operation* conv_op) {

    // --- 1. Extract operands and attributes ---
    if (conv_op->numOperands() < 2) {
        return std::unexpected(PassError{
            std::string(name()), 
            "Conv2D op has fewer than 2 operands"
        });
    }

    sir::Value* input  = conv_op->operand(0);  // [N, C, H, W]
    sir::Value* filter = conv_op->operand(1);  // [F, C, KH, KW]
    sir::Value* bias   = (conv_op->numOperands() > 2) ? conv_op->operand(2) : nullptr;

    auto strides   = conv_op->getAttrAs<std::vector<int64_t>>("strides").value_or(std::vector<int64_t>{1, 1});
    auto pads      = conv_op->getAttrAs<std::vector<int64_t>>("pads").value_or(std::vector<int64_t>{0, 0, 0, 0});
    auto dilations = conv_op->getAttrAs<std::vector<int64_t>>("dilations").value_or(std::vector<int64_t>{1, 1});

    const auto& in_dims  = input->shape().dims;
    const auto& fil_dims = filter->shape().dims;

    if (in_dims.size() != 4 || fil_dims.size() != 4) {
        return std::unexpected(PassError{
            std::string(name()), 
            "ConvLowering: expected rank-4 input and filter (NCHW)"
        });
    }

    const int64_t N  = in_dims[0];
    const int64_t C  = in_dims[1];
    const int64_t KH = fil_dims[2];
    const int64_t KW = fil_dims[3];
    const int64_t F  = fil_dims[0];

    // Reuse the shape formula
    auto conv_dim = [](int64_t in, int64_t k, int64_t s,
                       int64_t pb, int64_t pe, int64_t d) -> int64_t {
        if (in == sir::Shape::kDynamic) return sir::Shape::kDynamic;
        return (in + pb + pe - d * (k - 1) - 1) / s + 1;
    };

    const int64_t out_H = conv_dim(in_dims[2], KH, strides[0], pads[0], pads[2], dilations[0]);
    const int64_t out_W = conv_dim(in_dims[3], KW, strides[1], pads[1], pads[3], dilations[1]);

    const int64_t col_rows = C * KH * KW;
    const int64_t col_cols = (out_H == sir::Shape::kDynamic || out_W == sir::Shape::kDynamic) 
                                 ? sir::Shape::kDynamic 
                                 : out_H * out_W;

    // --- 2. Emit sc_low.im2col (Topology-Safe Insertion) ---
    auto im2col_op = block.insertOpBefore("sc_low.im2col", conv_op);
    im2col_op->addOperand(input);
    im2col_op->setAttribute("kernel_shape", std::vector<int64_t>{KH, KW});
    im2col_op->setAttribute("strides",   strides);
    im2col_op->setAttribute("pads",      pads);
    im2col_op->setAttribute("dilations", dilations);
    
    sir::Value* col_matrix = im2col_op->addResult("", input->dtype(), sir::Shape{{N, col_rows, col_cols}});

    // --- 3. Emit sc_low.view_cast (0-Cost Metadata update for Filter) ---
    auto view_filter_op = block.insertOpBefore("sc_low.view_cast", conv_op);
    view_filter_op->addOperand(filter);
    view_filter_op->setAttribute("target_shape", std::vector<int64_t>{F, col_rows});
    
    sir::Value* filter_matrix = view_filter_op->addResult("", filter->dtype(), sir::Shape{{F, col_rows}});

    // --- 4. Emit sc_low.matmul ---
    auto matmul_op = block.insertOpBefore("sc_low.matmul", conv_op);
    matmul_op->addOperand(filter_matrix);
    matmul_op->addOperand(col_matrix);

    sir::Value* matmul_out = matmul_op->addResult("", input->dtype(), sir::Shape{{N, F, col_cols}});

    // --- 5. Optionally emit sc_low.add for bias ---
    sir::Value* lowered_out = matmul_out;
    if (bias) {
        auto bias_op = block.insertOpBefore("sc_low.add", conv_op);
        bias_op->addOperand(matmul_out);
        bias_op->addOperand(bias);
        lowered_out = bias_op->addResult("", matmul_out->dtype(), matmul_out->shape());
    }

    // --- 6. Emit sc_low.view_cast (Spatial Recovery to 4D) ---
    // This is mandatory so downstream operations (ReLU, Pooling) receive 
    // the expected [N, F, H, W] layout rather than [N, F, H*W].
    auto view_out_op = block.insertOpBefore("sc_low.view_cast", conv_op);
    view_out_op->addOperand(lowered_out);
    view_out_op->setAttribute("target_shape", std::vector<int64_t>{N, F, out_H, out_W});
    
    sir::Value* final_4d_out = view_out_op->addResult("", lowered_out->dtype(), sir::Shape{{N, F, out_H, out_W}});

    // --- 7. Wire conv result users to the new 4D lowered output ---
    if (conv_op->numResults() > 0) {
        conv_op->result(0)->replaceAllUsesWith(final_4d_out);
    }

    // --- 8. Erase the original sc_high.conv2d ---
    block.removeOp(conv_op);

    utility::Logger::info(std::format("ConvLoweringPass: conv2d -> im2col + view_cast + matmul{}", 
                                      bias ? " + add(bias)" : ""));

    return {};
}

} // namespace seecpp::middle
