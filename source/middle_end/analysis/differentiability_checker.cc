#include "source/middle_end/analysis/differentiability_checker.h"

#include <string_view>
#include <unordered_set>

#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::analysis {

namespace {

// In a Google codebase, this would be an absl::flat_hash_set for better 
// cache locality, but std::unordered_set suffices for the standard library.
const std::unordered_set<std::string_view>& GetDiscreteOps() {
  static const auto* const kDiscreteOps =
      new std::unordered_set<std::string_view>{
          "ArgMax", "ArgMin", "NonZero", "Sign",
          "Floor",  "Ceil",   "Round",   "IsNaN",
          "IsInf",  "Equal",  "Greater", "Less"
      };
  return *kDiscreteOps;
}

// Registry of continuous operations that currently lack a registered 
// adjoint (gradient) kernel in the backend system.
const std::unordered_set<std::string_view>& GetMissingAdjoints() {
  static const auto* const kMissingAdjoints =
      new std::unordered_set<std::string_view>{
          // Example: Ops we haven't written the backward pass for yet.
          "Einsum", "DeformConv2D", "LpNormalization"
      };
  return *kMissingAdjoints;
}

}  // namespace

DifferentiabilityReport DifferentiabilityChecker::Analyze(
    const sir::Block& block) const {
  DifferentiabilityReport report;

  // Iterate over every operation in the IR block.
  for (const auto& op : block.operations()) {
    std::string_view mnemonic = op.mnemonic();

    if (IsDiscreteOp(mnemonic)) {
      report.violations.push_back({
          .node_name = op.name(),
          .op_mnemonic = std::string(mnemonic),
          .reason = "Operation is mathematically discrete and has no defined "
                    "gradient (violates Continuity constraint)."
      });
      continue;
    }

    if (IsMissingAdjoint(mnemonic)) {
      report.violations.push_back({
          .node_name = op.name(),
          .op_mnemonic = std::string(mnemonic),
          .reason = "Operation is continuous, but the Autodiff engine lacks "
                    "a registered adjoint (gradient) kernel."
      });
    }

    // Advanced Check (Placeholder): In a fully mature system, we would also 
    // check `op.operand_types()` here. For example, a "Cast" operation from 
    // Float32 to Int32 truncates the gradient flow and should be flagged if 
    // it lies on a path requiring gradients.
  }

  return report;
}

bool DifferentiabilityChecker::IsDiscreteOp(std::string_view mnemonic) const {
  const auto& discrete_ops = GetDiscreteOps();
  return discrete_ops.find(mnemonic) != discrete_ops.end();
}

bool DifferentiabilityChecker::IsMissingAdjoint(
    std::string_view mnemonic) const {
  const auto& missing_adjoints = GetMissingAdjoints();
  return missing_adjoints.find(mnemonic) != missing_adjoints.end();
}

}  // namespace seecpp::middle_end::analysis
