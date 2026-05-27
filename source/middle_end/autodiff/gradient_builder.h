#ifndef SEECPP_MIDDLE_END_AUTODIFF_GRADIENT_BUILDER_H_
#define SEECPP_MIDDLE_END_AUTODIFF_GRADIENT_BUILDER_H_

#include <string>
#include <string_view>
#include <unordered_map>

#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::autodiff {

/// @brief Manages the mapping between Primal nodes and their Adjoint (gradient)
/// nodes during the reverse pass. Handles automatic gradient accumulation.
class AdjointEnvironment {
 public:
  AdjointEnvironment() = default;

  /// @brief Accumulates a newly computed partial gradient for a primal node.
  /// If a gradient already exists (fan-out), it injects an 'Add' operation
  /// into the block to satisfy the multivariable chain rule.
  void Accumulate(std::string_view primal_name, std::string_view grad_name,
                  sir::Block& block);

  /// @brief Retrieves the current accumulated gradient for a primal node.
  /// @return The name of the gradient node, or empty if no gradient exists.
  [[nodiscard]] std::string GetGradient(std::string_view primal_name) const;

 private:
  // Maps a primal node name to its current accumulated gradient node name.
  std::unordered_map<std::string, std::string> adjoint_map_;
  
  // Counter to ensure unique names for auto-generated accumulation nodes.
  size_t accumulation_counter_ = 0;
};

/// @brief The Automatic Differentiation engine. Expands a static forward
/// SIR graph into a fully differentiable training graph using Reverse-Mode AD.
class GradientBuilder {
 public:
  GradientBuilder() = default;
  ~GradientBuilder() = default;

  GradientBuilder(const GradientBuilder&) = delete;
  GradientBuilder& operator=(const GradientBuilder&) = delete;

  /// @brief Wweaves the backward pass into the existing SIR block.
  /// @param block The graph to expand (must be topologically sorted).
  /// @param loss_node The scalar output node from which to begin the reverse pass.
  /// @return True if gradients were successfully built, false on unregistered ops.
  [[nodiscard]] bool BuildGradients(sir::Block& block,
                                    std::string_view loss_node);

 private:
  /// @brief Injects the base gradient seed (dL/dL = 1.0) into the block.
  std::string InjectLossSeed(sir::Block& block, std::string_view loss_node);
};

}  // namespace seecpp::middle_end::autodiff

#endif  // SEECPP_MIDDLE_END_AUTODIFF_GRADIENT_BUILDER_H_
