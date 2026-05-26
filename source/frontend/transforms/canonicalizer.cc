#include "source/transforms/canonicalizer.h"

#include <cassert>
#include <utility>

#include "seecpp/sir/sir.h"

namespace seecpp::transforms {

// --- PatternRewriter Implementation ---

void PatternRewriter::SetInsertionPoint(sir::Operation* op) {
  ip_ = op;
}

sir::Operation* PatternRewriter::CreateOp(
    std::string_view mnemonic,
    const std::vector<sir::Value*>& operands,
    const std::vector<sir::DataType>& result_types,
    const std::vector<sir::Shape>& result_shapes) {
  
  auto new_op = std::make_unique<sir::Operation>(std::string(mnemonic));
  
  for (sir::Value* op_val : operands) {
    new_op->addOperand(op_val);
  }

  assert(result_types.size() == result_shapes.size() && 
         "Result types and shapes must align.");

  for (size_t i = 0; i < result_types.size(); ++i) {
    // Generate automatic internal names for generated intermediate values
    std::string res_name = std::string(mnemonic) + "_out_" + std::to_string(i);
    new_op->addResult(res_name, result_types[i], result_shapes[i]);
  }

  // insertBefore must pass ownership to the block list and return a raw ptr
  return block_->insertBefore(ip_, std::move(new_op));
}

void PatternRewriter::ReplaceOp(sir::Operation* old_op,
                                const std::vector<sir::Value*>& new_values) {
  assert(old_op->getNumResults() == new_values.size() &&
         "Replacement values must match the number of original results.");

  for (size_t i = 0; i < new_values.size(); ++i) {
    sir::Value* old_res = old_op->getResult(i);
    
    // Replace all downstream uses of the old result with the new value
    // This requires a robust use-def chain implementation in sir::Value
    old_res->replaceAllUsesWith(new_values[i]);
  }

  EraseOp(old_op);
}

void PatternRewriter::EraseOp(sir::Operation* op) {
  // Unlink from operands to remove use-def chain references
  op->dropAllReferences();
  
  // Delegate to the block to deallocate and remove from the linked list
  block_->eraseOp(op);
}

// --- Canonicalizer Implementation ---

void Canonicalizer::AddPattern(std::unique_ptr<RewritePattern> pattern) {
  std::string target(pattern->GetTargetMnemonic());
  patterns_[std::move(target)].push_back(std::move(pattern));
}

bool Canonicalizer::RunOnBlock(sir::Block* block) {
  bool graph_changed = false;
  bool changed_this_iteration = true;
  int iteration = 0;

  PatternRewriter rewriter(block);

  // Fixed-point convergence loop
  while (changed_this_iteration && iteration < kMaxIterations) {
    changed_this_iteration = false;
    
    // We use a safe iterator paradigm. Because operations may be erased 
    // during MatchAndRewrite, we must pre-advance the iterator.
    auto it = block->begin();
    while (it != block->end()) {
      sir::Operation* op = &(*it);
      ++it; // Pre-advance

      auto pattern_it = patterns_.find(op->getMnemonic());
      if (pattern_it == patterns_.end()) {
        continue;
      }

      // Attempt to apply registered patterns for this operation type
      for (const auto& pattern : pattern_it->second) {
        rewriter.SetInsertionPoint(op);
        
        if (pattern->MatchAndRewrite(op, rewriter)) {
          changed_this_iteration = true;
          graph_changed = true;
          break; // Stop evaluating patterns for this op; it was modified/erased
        }
      }
    }
    
    iteration++;
  }

  return graph_changed;
}

}  // namespace seecpp::transforms
