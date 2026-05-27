#ifndef SEECPP_MIDDLE_END_TRANSFORMS_ALGEBRAIC_SIMPLIFIER_H_
#define SEECPP_MIDDLE_END_TRANSFORMS_ALGEBRAIC_SIMPLIFIER_H_

#include <string_view>

#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::transforms {

/// @brief Executes peephole optimizations on the SIR graph.
/// Applies algebraic identities and strength reduction to simplify the
/// computational graph for optimal backend performance.
class AlgebraicSimplifier {
 public:
  AlgebraicSimplifier() = default;
  ~AlgebraicSimplifier() = default;

  AlgebraicSimplifier(const AlgebraicSimplifier&) = delete;
  AlgebraicSimplifier& operator=(const AlgebraicSimplifier&) = delete;

  /// @brief Runs the algebraic simplification pass over the block until it
  /// reaches a fixed point (convergence).
  /// @param block The SIR block to optimize.
  /// @return True if the block was modified at all, false otherwise.
  bool Run(sir::Block& block);

 private:
  // Pattern matchers that return true if a transformation was applied.
  bool TrySimplifyAdd(sir::Operation& op, sir::Block& block);
  bool TrySimplifyMul(sir::Operation& op, sir::Block& block);
  bool TrySimplifyPow(sir::Operation& op, sir::Block& block);

  /// @brief Utility to check if a node represents a specific scalar constant.
  /// This interfaces with the constant environment to verify static values.
  [[nodiscard]] bool IsScalarConstant(std::string_view node_name,
                                      const sir::Block& block,
                                      float expected_value) const;
};

}  // namespace seecpp::middle_end::transforms

#endif  // SEECPP_MIDDLE_END_TRANSFORMS_ALGEBRAIC_SIMPLIFIER_H_
