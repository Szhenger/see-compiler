#include "source/middle_end/transforms/lowering/conv_lowering.h"

#include <format>
#include <vector>

#include "include/utility/logger.hpp"

namespace seecpp::middle_end::transforms::lowering {

namespace {
constexpr std::string_view kOpConv2d = "sc_high.conv2d";
constexpr std::string_view kOpConv2dGradInput = "sc_high.conv2d_grad_input";
constexpr std::string_view kOpConv2dGradFilter = "sc_high.conv2d_grad_filter";
}

bool ConvLowering::Run(sir::Block& block) {
  std::vector<sir::Operation*> to_lower;
  
  // Safely collect operations to avoid iterator invalidation during mutation
  block.walk([&](sir::Operation* op) {
    std::string_view mnem = op->mnemonic();
    if (mnem == kOpConv2d || mnem == kOpConv2dGradInput || mnem == kOpConv2dGradFilter) {
      to_lower.push_back(op);
    }
  });

  if (to_lower.empty()) return false;

  bool changed = false;
  for (sir::Operation* op : to_lower) {
    std::string_view mnem = op->mnemonic();
    if (mnem == kOpConv2d) {
      changed |= LowerForward(block, op);
    } else if (mnem == kOpConv2dGradInput) {
      changed |= LowerBackwardInput(block, op);
    } else if (mnem == kOpConv2dGradFilter) {
      changed |= LowerBackwardFilter(block, op);
    }
  }

  return changed;
}

// ... (LowerForward remains essentially identical to your excellent implementation) ...

bool ConvLowering::LowerBackwardInput(sir::Block& block, sir::Operation* op) {
  // Autodiff provides: [Filter Weights, Gradient Output]
  // We need to compute: Gradient Input (sc_low.col2im)
  if (op->numOperands() < 2) {
    if (diags_) {
      diags_->Report(op->location(), diagnostics::Level::Fatal)
          << "Conv2D Grad Input requires 2 operands (filter, grad_output).";
    }
    return false;
  }

  sir::Value* filter = op->operand(0);      // [F, C, KH, KW]
  sir::Value* grad_out = op->operand(1);    // [N, F, out_H, out_W]
  
  auto input_shape = op->getAttrAs<std::vector<int64_t>>("input_shape").value();
  auto strides = op->getAttrAs<std::vector<int64_t>>("strides").value();
  auto pads = op->getAttrAs<std::vector<int64_t>>("pads").value();

  const int64_t N = input_shape[0];
  const int64_t C = input_shape[1];
  const int64_t H = input_shape[2];
  const int64_t W = input_shape[3];

  const auto& fil_dims = filter->shape().dims;
  const int64_t F = fil_dims[0];
  const int64_t KH = fil_dims[2];
  const int64_t KW = fil_dims[3];

  const int64_t col_rows = C * KH * KW;
  const int64_t out_H = grad_out->shape().dims[2];
  const int64_t out_W = grad_out->shape().dims[3];
  const int64_t col_cols = out_H * out_W;

  // 1. Flatten Filter: [F, C, KH, KW] -> [F, C*KH*KW]
  auto view_filter = block.insertOpBefore("sc_low.view_cast", op);
  view_filter->addOperand(filter);
  view_filter->setAttribute("target_shape", std::vector<int64_t>{F, col_rows});
  sir::Value* flat_filter = view_filter->addResult("", filter->dtype(), sir::Shape{{F, col_rows}});

  // 2. Transpose Filter: [F, C*KH*KW] -> [C*KH*KW, F]
  auto trans_filter = block.insertOpBefore("sc_low.transpose", op);
  trans_filter->addOperand(flat_filter);
  sir::Value* filter_T = trans_filter->addResult("", filter->dtype(), sir::Shape{{col_rows, F}});

  // 3. Flatten GradOut: [N, F, out_H, out_W] -> [N, F, out_H * out_W]
  auto view_grad = block.insertOpBefore("sc_low.view_cast", op);
  view_grad->addOperand(grad_out);
  view_grad->setAttribute("target_shape", std::vector<int64_t>{N, F, col_cols});
  sir::Value* flat_grad = view_grad->addResult("", grad_out->dtype(), sir::Shape{{N, F, col_cols}});

  // 4. MatMul: Filter^T * GradOut = [N, C*KH*KW, out_H*out_W]
  // This yields the scattered gradients in column space.
  auto matmul_op = block.insertOpBefore("sc_low.matmul", op);
  matmul_op->addOperand(filter_T);
  matmul_op->addOperand(flat_grad);
  sir::Value* col_grad = matmul_op->addResult("", grad_out->dtype(), sir::Shape{{N, col_rows, col_cols}});

  // 5. Col2Im: Accumulate column gradients back into spatial image layout [N, C, H, W]
  auto col2im_op = block.insertOpBefore("sc_low.col2im", op);
  col2im_op->addOperand(col_grad);
  col2im_op->setAttribute("target_shape", input_shape);
  col2im_op->setAttribute("kernel_shape", std::vector<int64_t>{KH, KW});
  col2im_op->setAttribute("strides", strides);
  col2im_op->setAttribute("pads", pads);
  
  sir::Value* final_grad_in = col2im_op->addResult("", grad_out->dtype(), sir::Shape{input_shape});

  // 6. Wire and cleanup
  op->result(0)->replaceAllUsesWith(final_grad_in);
  block.removeOp(op);

  utility::Logger::debug("ConvLowering: Lowered grad_input -> transpose + matmul + col2im");
  return true;
}

bool ConvLowering::LowerBackwardFilter(sir::Block& block, sir::Operation* op) {
  // Autodiff provides: [Forward Input, Gradient Output]
  // We need to compute: Gradient Filter (im2col -> matmul)
  
  sir::Value* input = op->operand(0);       // [N, C, H, W]
  sir::Value* grad_out = op->operand(1);    // [N, F, out_H, out_W]

  // ... Extract dimensions ... (omitted for brevity, matches above) ...

  // 1. im2col on Forward Input (re-materialize the forward column matrix)
  auto im2col_op = block.insertOpBefore("sc_low.im2col", op);
  im2col_op->addOperand(input);
  // ... set attributes ...
  sir::Value* col_matrix = im2col_op->addResult("", input->dtype(), sir::Shape{{N, col_rows, col_cols}});

  // 2. Transpose ColMatrix: [N, C*KH*KW, out_H*out_W] -> [N, out_H*out_W, C*KH*KW]
  auto trans_col = block.insertOpBefore("sc_low.transpose", op);
  trans_col->addOperand(col_matrix);
  sir::Value* col_matrix_T = trans_col->addResult("", input->dtype(), sir::Shape{{N, col_cols, col_rows}});

  // 3. Flatten GradOut: [N, F, out_H, out_W] -> [N, F, out_H*out_W]
  auto view_grad = block.insertOpBefore("sc_low.view_cast", op);
  view_grad->addOperand(grad_out);
  // ...
  sir::Value* flat_grad = view_grad->addResult("", grad_out->dtype(), sir::Shape{{N, F, col_cols}});

  // 4. Batched MatMul: GradOut * ColMatrix^T = [N, F, C*KH*KW]
  auto matmul_op = block.insertOpBefore("sc_low.matmul", op);
  matmul_op->addOperand(flat_grad);
  matmul_op->addOperand(col_matrix_T);
  sir::Value* batch_grad_filter = matmul_op->addResult("", grad_out->dtype(), sir::Shape{{N, F, col_rows}});

  // 5. ReduceSum across Batch (N) dimension -> [F, C*KH*KW]
  auto reduce_op = block.insertOpBefore("sc_low.reduce_sum", op);
  reduce_op->addOperand(batch_grad_filter);
  reduce_op->setAttribute("axis", std::vector<int64_t>{0});
  sir::Value* flat_grad_filter = reduce_op->addResult("", grad_out->dtype(), sir::Shape{{F, col_rows}});

  // 6. ViewCast back to 4D Filter Shape -> [F, C, KH, KW]
  auto view_out = block.insertOpBefore("sc_low.view_cast", op);
  view_out->addOperand(flat_grad_filter);
  view_out->setAttribute("target_shape", std::vector<int64_t>{F, C, KH, KW});
  sir::Value* final_grad_filter = view_out->addResult("", grad_out->dtype(), sir::Shape{{F, C, KH, KW}});

  op->result(0)->replaceAllUsesWith(final_grad_filter);
  block.removeOp(op);

  utility::Logger::debug("ConvLowering: Lowered grad_filter -> im2col + matmul + reduce_sum");
  return true;
}

}  // namespace seecpp::middle_end::transforms::lowering
