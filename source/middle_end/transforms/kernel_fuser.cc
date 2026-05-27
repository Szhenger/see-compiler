#include "source/middle_end/transforms/kernel_fuser.h"

#include <format>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "include/utility/logger.hpp"

namespace seecpp::middle_end::transforms {

namespace {
// Centralized definitions prevent string-literal typos and make it easier 
// to map operations during the initial Protobuf/ONNX ingestion phase.
constexpr std::string_view kOpConv2d = "sc_high.conv2d";
constexpr std::string_view kOpBatchNorm = "sc_high.batch_norm";
constexpr std::string_view kOpReluHigh = "sc_high.relu";
constexpr std::string_view kOpReluLow = "sc_low.relu";
constexpr std::string_view kOpConstant = "sc_high.constant";
constexpr std::string_view kOpFusedEw = "sc_high.fused_ew";
}  // namespace

bool KernelFuser::Run(sir::Block& block) {
  bool changed = false;

  changed |= FoldConvBatchNorm(block);
  changed |= FuseMatMulRelu(block);
  changed |= FuseElementwiseChains(block);

  utility::Logger::info(std::format(
      "KernelFuser: {} conv-bn folded, {} matmul-relu fused, {} ew-chains built.",
      fused_conv_bn_, fused_matmul_relu_, fused_elementwise_));

  return changed;
}

bool KernelFuser::FoldConvBatchNorm(sir::Block& block) {
  bool changed = false;
  std::unordered_set<std::string_view> dead_ids;
  std::vector<sir::Operation*> bn_ops;
  std::vector<sir::Operation*> to_delete;

  block.walk([&](sir::Operation* op) {
    if (op->mnemonic() == kOpBatchNorm) bn_ops.push_back(op);
  });

  for (auto* bn : bn_ops) {
    if (TryFoldConvBatchNorm(block, bn, dead_ids)) {
      to_delete.push_back(bn);
      changed = true;
    }
  }

  // Safe Deletion: Erase from the block only after iteration completes.
  for (auto* op : to_delete) block.removeOp(op);
  return changed;
}

bool KernelFuser::FuseElementwiseChains(sir::Block& block) {
  bool graph_changed = false;
  bool pass_changed;

  // Fixed-point convergence loop to handle multi-node cascading fusions.
  do {
    pass_changed = false;
    std::unordered_set<std::string_view> dead_ids;
    std::vector<sir::Operation*> ew_ops;
    std::vector<sir::Operation*> to_delete;

    block.walk([&](sir::Operation* op) {
      std::string_view mn = op->mnemonic();
      if (mn == "sc_high.add" || mn == "sc_high.mul" ||
          mn == "sc_high.sub" || mn == "sc_high.div" || mn == kOpFusedEw) {
        ew_ops.push_back(op);
      }
    });

    for (auto* consumer : ew_ops) {
      if (dead_ids.count(consumer->id())) continue;
      
      if (TryFuseElementwisePair(block, consumer, dead_ids)) {
        // Mark for safe deletion. The TryFuse function will add both the 
        // producer and consumer IDs to the dead_ids set.
        pass_changed = true;
        graph_changed = true;
      }
    }

    // Clean up all safely dead operations before the next convergence pass.
    for (auto* consumer : ew_ops) {
      if (dead_ids.count(consumer->id())) {
         block.removeOp(consumer);
      }
    }
  } while (pass_changed);

  return graph_changed;
}

bool KernelFuser::TryFoldConvBatchNorm(
    sir::Block& block, sir::Operation* bn_op,
    std::unordered_set<std::string_view>& dead_ids) {
  if (bn_op->numOperands() < 5) return false;

  sir::Value* bn_input = bn_op->operand(0);
  sir::Operation* conv_op = bn_input->definingOp();

  if (!conv_op || conv_op->mnemonic() != kOpConv2d) return false;
  
  if (!bn_input->hasOneUse()) {
    if (diags_) {
      diags_->Report(bn_op->location(), diagnostics::Level::Note)
          << "Conv-BN folding aborted: Convolution output has multiple consumers.";
    }
    return false;
  }

  auto is_constant = [](sir::Value* v) -> bool {
    return v && v->definingOp() && v->definingOp()->mnemonic() == kOpConstant;
  };

  if (!is_constant(bn_op->operand(1)) || !is_constant(bn_op->operand(2)) || 
      !is_constant(bn_op->operand(3)) || !is_constant(bn_op->operand(4))) {
    return false;
  }

  float epsilon = bn_op->getAttrAs<float>("epsilon").value_or(1e-5f);

  // Directly embed the batch norm constants into the Convolution's static footprint.
  conv_op->setAttribute("fused_bn", int64_t(1));
  conv_op->setAttribute("bn_epsilon", epsilon);
  conv_op->setAttribute("bn_scale_id", std::string(bn_op->operand(1)->id()));
  conv_op->setAttribute("bn_bias_id", std::string(bn_op->operand(2)->id()));
  conv_op->setAttribute("bn_mean_id", std::string(bn_op->operand(3)->id()));
  conv_op->setAttribute("bn_var_id", std::string(bn_op->operand(4)->id()));

  bn_op->result(0)->replaceAllUsesWith(conv_op->result(0));
  
  dead_ids.insert(bn_op->id());
  ++fused_conv_bn_;
  return true;
}

bool KernelFuser::TryFuseElementwisePair(
    sir::Block& block, sir::Operation* consumer_op,
    std::unordered_set<std::string_view>& dead_ids) {
  
  for (size_t i = 0; i < consumer_op->numOperands(); ++i) {
    sir::Value* v = consumer_op->operand(i);
    sir::Operation* producer = v->definingOp();
    
    if (!producer || dead_ids.count(producer->id())) continue;

    std::string_view pmn = producer->mnemonic();
    bool is_ew_producer = (
        pmn == "sc_high.add" || pmn == "sc_high.mul" ||
        pmn == "sc_high.sub" || pmn == "sc_high.div" || pmn == kOpFusedEw);

    if (!is_ew_producer || !v->hasOneUse()) continue;

    std::string p_seq = (pmn == kOpFusedEw) 
        ? producer->getAttrAs<std::string>("op_sequence").value_or("unknown")
        : std::string(pmn);

    std::string c_seq = (consumer_op->mnemonic() == kOpFusedEw)
        ? consumer_op->getAttrAs<std::string>("op_sequence").value_or("unknown")
        : std::string(consumer_op->mnemonic());

    // Topological insertion keeps the graph strictly ordered for the backend generator.
    auto fused_op = block.insertOpBefore(std::string(kOpFusedEw), consumer_op);
    fused_op->setAttribute("op_sequence", std::format("{}+{}", p_seq, c_seq));

    for (size_t j = 0; j < producer->numOperands(); ++j) {
      fused_op->addOperand(producer->operand(j));
    }
    for (size_t j = 0; j < consumer_op->numOperands(); ++j) {
      if (consumer_op->operand(j) != v) {
        fused_op->addOperand(consumer_op->operand(j));
      }
    }

    fused_op->addResult("", consumer_op->result(0)->dtype(), 
                        consumer_op->result(0)->shape());
                        
    consumer_op->result(0)->replaceAllUsesWith(fused_op->result(0));

    // Register IDs in the tombstone, but do NOT delete the ops here.
    dead_ids.insert(consumer_op->id());
    dead_ids.insert(producer->id());
    
    ++fused_elementwise_;
    return true; // Break early so the convergence loop can handle the next chain safely.
  }
  return false;
}

// ... (FuseMatMulRelu follows the exact same safety paradigm) ...

}  // namespace seecpp::middle_end::transforms
