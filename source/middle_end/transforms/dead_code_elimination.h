#ifndef SEECPP_MIDDLE_END_TRANSFORMS_DEAD_CODE_ELIMINATION_H_
#define SEECPP_MIDDLE_END_TRANSFORMS_DEAD_CODE_ELIMINATION_H_

#include "seecpp/diagnostics/diagnostics_engine.h"
#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::transforms {

/// @brief Removes unreachable or unused operations using a backward 
/// mark-and-sweep algorithm.
class DeadCodeElimination {
 public:
  explicit DeadCodeElimination(diagnostics::DiagnosticsEngine* diags = nullptr)
      : diags_(diags) {}
  ~DeadCodeElimination() = default;

  DeadCodeElimination(const DeadCodeElimination&) = delete;
  DeadCodeElimination& operator=(const DeadCodeElimination&) = delete;

  /// @brief Executes the DCE pass over the block.
  /// @param block The SIR block to clean.
  /// @return True if any dead operations were removed.
  bool Run(sir::Block& block);

 private:
  /// @brief Determines if an operation is a mandatory root (e.g., returns, 
  /// memory stores, or side-effects) that cannot be eliminated.
  [[nodiscard]] bool IsRootOperation(const sir::Operation* op) const;

  diagnostics::DiagnosticsEngine* diags_;
};

}  // namespace seecpp::middle_end::transforms

#endif  // SEECPP_MIDDLE_END_TRANSFORMS_DEAD_CODE_ELIMINATION_H_
