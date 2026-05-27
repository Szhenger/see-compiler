#include "source/middle_end/transforms/algebraic_simplifier.h"

#include <string>
#include <string_view>
#include <vector>

#include "include/utility/logger.hpp"
#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::transforms {

bool AlgebraicSimplifier::Run(sir::Block& block) {
  bool graph_changed = false;
  bool pass_changed;

  // Fixed-point iteration: continue applying rules until the graph converges.
  // This ensures cascaded optimizations (e.g., (x * 1) + 0 -> x + 0 -> x)
  // are fully resolved.
  do {
    pass_changed = false;
    
    // Snapshot the operations to safely iterate. Modifying the block directly
    // during traversal would invalidate standard iterators.
    std::vector<sir::Operation> current_ops = block.operations();

    for (auto& op : current_ops) {
      // If the operation was already erased in this specific pass, skip it.
      if (!block.HasOperation(op.name())) continue;

      std::string_view mnemonic = op.mnemonic();
      bool op_changed = false;

      if (mnemonic == "Add") {
        op_changed = TrySimplifyAdd(op, block);
      } else if (mnemonic == "Mul") {
        op_changed = TrySimplifyMul(op, block);
      } else if (mnemonic == "Pow") {
        op_changed = TrySimplifyPow(op, block);
      }

      if (op_changed) {
        pass_changed = true;
        graph_changed = true;
      }
    }
  } while (pass_changed);

  if (graph_changed) {
    utility::Logger::debug("AlgebraicSimplifier: Optimization converged.");
  }
  
  return graph_changed;
}

bool AlgebraicSimplifier::TrySimplifyAdd(sir::Operation& op,
                                         sir::Block& block) {
  const auto& operands = op.operand_names();
  if (operands.size() != 2) return false;

  std::string lhs = operands[0];
  std::string rhs = operands[1];
  std::string simplified_node;

  // Identity Removal: x + 0 -> x
  if (IsScalarConstant(rhs, block, 0.0f)) {
    simplified_node = lhs;
  } else if (IsScalarConstant(lhs, block, 0.0f)) {
    simplified_node = rhs;
  }

  if (!simplified_node.empty()) {
    // Rewire all downstream consumers to point to the simplified node,
    // then safely prune the redundant addition from the graph.
    block.ReplaceAllUsesWith(op.name(), simplified_node);
    block.EraseOperation(op.name());
    return true;
  }
  
  return false;
}

bool AlgebraicSimplifier::TrySimplifyMul(sir::Operation& op,
                                         sir::Block& block) {
  const auto& operands = op.operand_names();
  if (operands.size() != 2) return false;

  std::string lhs = operands[0];
  std::string rhs = operands[1];
  std::string simplified_node;

  // Identity Removal: x * 1 -> x
  if (IsScalarConstant(rhs, block, 1.0f)) {
    simplified_node = lhs;
  } else if (IsScalarConstant(lhs, block, 1.0f)) {
    simplified_node = rhs;
  }
  // Zero Property: x * 0 -> 0
  else if (IsScalarConstant(rhs, block, 0.0f)) {
    simplified_node = rhs;
  } else if (IsScalarConstant(lhs, block, 0.0f)) {
    simplified_node = lhs;
  }

  if (!simplified_node.empty()) {
    block.ReplaceAllUsesWith(op.name(), simplified_node);
    block.EraseOperation(op.name());
    return true;
  }
  
  return false;
}

bool AlgebraicSimplifier::TrySimplifyPow(sir::Operation& op,
                                         sir::Block& block) {
  const auto& operands = op.operand_names();
  if (operands.size() != 2) return false;

  std::string base = operands[0];
  std::string exp = operands[1];

  // Strength Reduction: x^2 -> x * x
  if (IsScalarConstant(exp, block, 2.0f)) {
    // Replace the expensive Power operation with a cheap Multiply operation.
    // We reuse the name of the Pow node so downstream dependencies naturally
    // pick up the new instruction without needing a full Re-Use wiring.
    sir::Operation new_mul =
        sir::Operation::Create(op.name(), "Mul", {base, base});
    block.ReplaceOperation(op.name(), new_mul);
    return true;
  }

  // Identity Removal: x^1 -> x
  if (IsScalarConstant(exp, block, 1.0f)) {
    block.ReplaceAllUsesWith(op.name(), base);
    block.EraseOperation(op.name());
    return true;
  }

  return false;
}

bool AlgebraicSimplifier::IsScalarConstant(std::string_view node_name,
                                           const sir::Block& block,
                                           float expected_value) const {
  const sir::Operation* producer = block.GetOperation(std::string(node_name));
  
  // To match a pattern, the producing node must exist and be a 'Constant'
  if (!producer || producer->mnemonic() != "Constant") {
    return false;
  }

  // NOTE: This assumes the 'Constant' operation in the SIR supports querying
  // its payload. In a fully wired compiler, this might bridge to the
  // previously built `ConstantFolder` or `WeightBuffer`.
  return producer->GetScalarValueAsFloat() == expected_value;
}

}  // namespace seecpp::middle_end::transforms
