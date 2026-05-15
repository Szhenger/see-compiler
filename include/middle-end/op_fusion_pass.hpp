#pragma once

#include "middle-end/pass_manager.hpp"
#include "middle-end/sir.hpp"

namespace seecpp::middle {

enum class FusionScope : uint8_t {
    HighLevelDialect,
    LowLevelDialect
};

/**
 * @brief Fuses adjacent operations to maximize register re-use and eliminate 
 * memory round-trips to the Static Arena.
 * 
 * DESIGN PRINCIPLES:
 * 1. Scope Isolation: Parameterized by FusionScope to cleanly execute target patterns 
 *    pre- or post-lowering without logic pollution.
 * 2. Pure Transformations: Eliminates internal counter state in favor of return metrics 
 *    or structural logging to maintain pipeline thread-safety.
 * 3. Use-Def Chain Validation: Ensures producer ops have a strict single-user contract 
 *    before executing destructive structural composition.
 */

class OperatorFusionPass final : public IPass {
public:
    explicit OperatorFusionPass(FusionScope scope) : scope_(scope) {}

    std::string_view name() const override {
        return scope_ == FusionScope::HighLevelDialect ? "sc_high.operator_fusion" 
                                                       : "sc_low.operator_fusion";
    }

    bool requiresValidation() const override { return true; }

    [[nodiscard]]
    PassResult runOnBlock(sir::Block& block) override;

private:
    [[nodiscard]] PassResult runHighLevelFusion(sir::Block& block);
    [[nodiscard]] PassResult runLowLevelFusion(sir::Block& block);

    bool foldConvBatchNorm(sir::Block& block, sir::Operation* bn_op);
    bool foldMatMulRelu(sir::Block& block, sir::Operation* relu_op);
    bool foldElementwiseChain(sir::Block& block, sir::Operation* consumer_op);

    FusionScope scope_;
};

} // namespace seecpp::middle
