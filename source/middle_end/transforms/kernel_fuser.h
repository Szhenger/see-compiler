#ifndef SEECPP_MIDDLE_END_TRANSFORMS_KERNEL_FUSER_H_
#define SEECPP_MIDDLE_END_TRANSFORMS_KERNEL_FUSER_H_

#include <string>
#include <string_view>
#include <unordered_map>

#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::transforms {

/// @brief Combines chains of element-wise operations into single fused kernels.
/// This drastically reduces memory bandwidth constraints by keeping intermediate
/// values in hardware registers.
class KernelFuser {
 public:
  KernelFuser() = default;
  ~KernelFuser() = default;

  KernelFuser(const KernelFuser&) = delete;
  KernelFuser& operator=(const KernelFuser&) = delete;

  /// @brief Executes the fusion pass over the block until convergence.
  /// @param block The SIR block to optimize.
  /// @return True if any operations were fused, false otherwise.
  bool Run(sir::Block& block);

 private:
  /// @brief Determines if an operator is purely element-wise (e.g., Add, Relu)
  /// and eligible for memory-bound kernel fusion.
  [[nodiscard]] bool IsElementwise(std::string_view mnemonic) const;

  /// @brief Performs a full scan of the block to compute the exact number of 
  /// downstream consumers for every node. We can only safely fuse an intermediate
  /// node if it has exactly 1 consumer to avoid recomputation overhead.
  [[nodiscard]] std::unordered_map<std::string, int> ComputeUseCounts(
      const sir::Block& block) const;

  /// @brief Attempts a single pairwise fusion between a producer and consumer.
  bool TryFusePair(sir::Operation& consumer, sir::Block& block,
                   const std::unordered_map<std::string, int>& use_counts);
};

}  // namespace seecpp::middle_end::transforms

#endif  // SEECPP_MIDDLE_END_TRANSFORMS_KERNEL_FUSER_H_
