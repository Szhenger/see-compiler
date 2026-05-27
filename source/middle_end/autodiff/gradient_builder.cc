#include "source/middle_end/autodiff/gradient_builder.h"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "include/utility/logger.hpp"
#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::autodiff {

namespace {

// A VJP Rule computes the partial gradients for an operator's inputs,
// given the incoming gradient from its output.
using VjpRule = std::function<bool(
    const sir::Operation& primal_op, std::string_view out_grad_name,
    sir::Block& block, AdjointEnvironment& env)>;

// The static registry of mathematical derivatives.
const std::unordered_map<std::string_view, VjpRule>& GetVjpRegistry() {
  static const auto* const kRegistry =
      new std::unordered_map<std::string_view, VjpRule>{
          
          // Rule for C = A + B
          // dL/dA = dL/dC * 1, dL/dB = dL/dC * 1
          {"Add", [](const sir::Operation& op, std::string_view out_grad,
                     sir::Block& block, AdjointEnvironment& env) {
            const auto& operands = op.operand_names();
            if (operands.size() != 2) return false;
            
            // Gradients route directly back to both inputs (broadcasting 
            // semantics would be handled by the VJP rule in a full system).
            env.Accumulate(operands[0], out_grad, block);
            env.Accumulate(operands[1], out_grad, block);
            return true;
          }},

          // Rule for C = MatMul(A, B)
          // dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
          {"MatMul", [](const sir::Operation& op, std::string_view out_grad,
                        sir::Block& block, AdjointEnvironment& env) {
            const auto& operands = op.operand_names();
            std::string a_name = operands[0];
            std::string b_name = operands[1];

            std::string a_grad = op.name() + "_grad_A";
            std::string b_grad = op.name() + "_grad_B";

            // Inject: dL/dA = MatMul(dL/dC, Transpose(B))
            block.AddOperation(sir::Operation::Create(
                a_grad, "MatMulGradA", {std::string(out_grad), b_name}));
            
            // Inject: dL/dB = MatMul(Transpose(A), dL/dC)
            block.AddOperation(sir::Operation::Create(
                b_grad, "MatMulGradB", {a_name, std::string(out_grad)}));

            env.Accumulate(a_name, a_grad, block);
            env.Accumulate(b_name, b_grad, block);
            return true;
          }},

          // Rule for Y = Relu(X)
          // dL/dX = dL/dY * (X > 0 ? 1 : 0)
          {"Relu", [](const sir::Operation& op, std::string_view out_grad,
                      sir::Block& block, AdjointEnvironment& env) {
            const auto& operands = op.operand_names();
            std::string x_name = operands[0];
            std::string x_grad = op.name() + "_grad_X";

            // Inject: dL/dX = ReluGrad(dL/dY, X)
            block.AddOperation(sir::Operation::Create(
                x_grad, "ReluGrad", {std::string(out_grad), x_name}));

            env.Accumulate(x_name, x_grad, block);
            return true;
          }}
      };
  return *kRegistry;
}

}  // namespace

void AdjointEnvironment::Accumulate(std::string_view primal_name,
                                    std::string_view grad_name,
                                    sir::Block& block) {
  std::string primal_str(primal_name);
  auto it = adjoint_map_.find(primal_str);

  if (it == adjoint_map_.end()) {
    // First time we are seeing a gradient for this primal node.
    adjoint_map_[primal_str] = std::string(grad_name);
  } else {
    // Fan-out detected: A gradient already exists. We must sum them.
    std::string existing_grad = it->second;
    std::string sum_name =
        primal_str + "_grad_acc_" + std::to_string(accumulation_counter_++);

    block.AddOperation(sir::Operation::Create(
        sum_name, "Add", {existing_grad, std::string(grad_name)}));

    // Update the environment to point to the new accumulated gradient.
    it->second = sum_name;
  }
}

std::string AdjointEnvironment::GetGradient(std::string_view primal_name) const {
  auto it = adjoint_map_.find(std::string(primal_name));
  return (it != adjoint_map_.end()) ? it->second : "";
}

std::string GradientBuilder::InjectLossSeed(sir::Block& block,
                                            std::string_view loss_node) {
  std::string seed_name = std::string(loss_node) + "_grad_seed";
  // Generates a tensor of 1.0s matching the shape of the loss node.
  block.AddOperation(sir::Operation::Create(
      seed_name, "OnesLike", {std::string(loss_node)}));
  return seed_name;
}

bool GradientBuilder::BuildGradients(sir::Block& block,
                                     std::string_view loss_node) {
  AdjointEnvironment env;
  const auto& registry = GetVjpRegistry();

  // 1. Seed the gradient for the loss node (dL/dL = 1.0).
  std::string loss_seed = InjectLossSeed(block, loss_node);
  env.Accumulate(loss_node, loss_seed, block);

  // 2. Snapshot the primal operations. We cannot iterate over block.operations()
  // directly because we are actively appending adjoint operations to it.
  const std::vector<sir::Operation> primal_ops = block.operations();

  // 3. Reverse topological traversal.
  for (auto it = primal_ops.rbegin(); it != primal_ops.rend(); ++it) {
    const sir::Operation& op = *it;

    // Check if this node has a downstream gradient. If it doesn't, it means
    // it does not contribute to the loss (Dead Code Elimination for Autodiff).
    std::string out_grad = env.GetGradient(op.name());
    if (out_grad.empty()) {
      continue;
    }

    // Parameters/Constants don't generate upstream gradients.
    if (op.mnemonic() == "Constant" || op.mnemonic() == "Variable") {
      continue;
    }

    auto rule_it = registry.find(op.mnemonic());
    if (rule_it == registry.end()) {
      utility::Logger::error(
          "GradientBuilder: Missing VJP rule for operator '" +
          std::string(op.mnemonic()) + "' at node '" + op.name() + "'.");
      return false;
    }

    // Execute the VJP rule to compute input gradients and update the environment.
    if (!rule_it->second(op, out_grad, block, env)) {
      utility::Logger::error(
          "GradientBuilder: VJP application failed for operator '" +
          std::string(op.mnemonic()) + "'.");
      return false;
    }
  }

  return true;
}

}  // namespace seecpp::middle_end::autodiff
