#ifndef SEECPP_MIDDLE_END_TRANSFORMS_LOWERING_CONV_LOWERING_H_
#define SEECPP_MIDDLE_END_TRANSFORMS_LOWERING_CONV_LOWERING_H_

#include <string_view>
#include <unordered_set>

#include "seecpp/diagnostics/diagnostics_engine.h"
#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::transforms::lowering {

/// @brief Lowers high-level spatial convolution operations (forward and backward)
/// into hardware-aligned linear algebra primitives (im2col, col2im, matmul).
class ConvLowering {
 public:
  explicit ConvLowering(diagnostics::DiagnosticsEngine* diags = nullptr)
      : diags_(diags) {}
  ~ConvLowering() = default;

  ConvLowering(const ConvLowering&) = delete;
  ConvLowering& operator=(const ConvLowering&) = delete;

  /// @brief Executes the lowering pass over the block.
  /// @return True if any operations were lowered.
  bool Run(sir::Block& block);

 private:
  bool LowerForward(sir::Block& block, sir::Operation* op);
  
  /// @brief Lowers the gradient with respect to the input activations.
  /// Requires sc_low.col2im to accumulate overlapping gradient windows.
  bool LowerBackwardInput(sir::Block& block, sir::Operation* op);

  /// @brief Lowers the gradient with respect to the filter weights.
  bool LowerBackwardFilter(sir::Block& block, sir::Operation* op);

  diagnostics::DiagnosticsEngine* diags_;
};

}  // namespace seecpp::middle_end::transforms::lowering

#endif  // SEECPP_MIDDLE_END_TRANSFORMS_LOWERING_CONV_LOWERING_H_
