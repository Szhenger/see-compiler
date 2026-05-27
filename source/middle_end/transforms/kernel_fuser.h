#ifndef SEECPP_MIDDLE_END_TRANSFORMS_KERNEL_FUSER_H_
#define SEECPP_MIDDLE_END_TRANSFORMS_KERNEL_FUSER_H_

#include <string_view>
#include <unordered_set>
#include <vector>

#include "seecpp/diagnostics/diagnostics_engine.h"
#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::transforms {

/// @brief Fuses compatible sub-graphs to maximize register locality and 
/// eliminate unnecessary global memory round-trips.
class KernelFuser {
 public:
  explicit KernelFuser(diagnostics::DiagnosticsEngine* diags = nullptr) 
      : diags_(diags) {}
  ~KernelFuser() = default;

  KernelFuser(const KernelFuser&) = delete;
  KernelFuser& operator=(const KernelFuser&) = delete;

  /// @brief Executes all fusion passes over the block.
  /// @param block The SIR block to optimize.
  /// @return True if any operations were fused.
  bool Run(sir::Block& block);

 private:
  // Sub-passes
  bool FoldConvBatchNorm(sir::Block& block);
  bool FuseMatMulRelu(sir::Block& block);
  bool FuseElementwiseChains(sir::Block& block);

  // Core fusion implementations
  bool TryFoldConvBatchNorm(sir::Block& block, sir::Operation* bn_op,
                            std::unordered_set<std::string_view>& dead_ids);
  bool TryFuseMatMulRelu(sir::Block& block, sir::Operation* relu_op,
                         std::unordered_set<std::string_view>& dead_ids);
  bool TryFuseElementwisePair(sir::Block& block, sir::Operation* consumer_op,
                              std::unordered_set<std::string_view>& dead_ids);

  // Optional diagnostics hook for terminal tracing
  diagnostics::DiagnosticsEngine* diags_;

  // Metrics
  size_t fused_conv_bn_ = 0;
  size_t fused_matmul_relu_ = 0;
  size_t fused_elementwise_ = 0;
};

}  // namespace seecpp::middle_end::transforms

#endif  // SEECPP_MIDDLE_END_TRANSFORMS_KERNEL_FUSER_H_
