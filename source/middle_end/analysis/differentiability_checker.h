#ifndef SEECPP_MIDDLE_END_ANALYSIS_DIFFERENTIABILITY_CHECKER_H_
#define SEECPP_MIDDLE_END_ANALYSIS_DIFFERENTIABILITY_CHECKER_H_

#include <string>
#include <vector>

#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::analysis {

/// @brief Represents a single violation of the Continuity constraint.
struct DifferentiabilityViolation {
  std::string node_name;
  std::string op_mnemonic;
  std::string reason;
};

/// @brief The result of the differentiability analysis pass.
struct DifferentiabilityReport {
  std::vector<DifferentiabilityViolation> violations;

  [[nodiscard]] bool IsDifferentiable() const { return violations.empty(); }
};

/// @brief Analyzes a SIR block to ensure all operations on the active path 
/// are mathematically continuous and support reverse-mode differentiation.
class DifferentiabilityChecker {
 public:
  DifferentiabilityChecker() = default;
  ~DifferentiabilityChecker() = default;

  // Prevent copying and moving for strict pass-manager integration.
  DifferentiabilityChecker(const DifferentiabilityChecker&) = delete;
  DifferentiabilityChecker& operator=(const DifferentiabilityChecker&) = delete;

  /// @brief Scans the block for discrete or non-differentiable operations.
  /// @param block The SIR graph to analyze.
  /// @return A report containing any violations found.
  [[nodiscard]] DifferentiabilityReport Analyze(const sir::Block& block) const;

 private:
  /// @brief Determines if a specific operator mnemonic is strictly discrete.
  [[nodiscard]] bool IsDiscreteOp(std::string_view mnemonic) const;

  /// @brief Determines if an operator is differentiable but currently lacks
  /// an implemented adjoint kernel in the Autodiff engine.
  [[nodiscard]] bool IsMissingAdjoint(std::string_view mnemonic) const;
};

}  // namespace seecpp::middle_end::analysis

#endif  // SEECPP_MIDDLE_END_ANALYSIS_DIFFERENTIABILITY_CHECKER_H_
