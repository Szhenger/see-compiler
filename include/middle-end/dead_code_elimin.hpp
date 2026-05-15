#pragma once

#include "middle-end/pass_manager.hpp"
#include "middle-end/sir.hpp"

namespace seecpp::middle {

/**
 * DeadCodeEliminationPass: High-efficiency SSA Pruning.
 * 
 * DESIGN PRINCIPLES:
 * 1. Side-Effect Awareness: Operations marked as having side-effects (e.g., Print, 
 *    State-Update) are never pruned regardless of user count.
 * 2. Recursive Worklist: When an op is deleted, its operands are added to a 
 *    worklist to check if they have become "newly dead," preventing residual clutter.
 * 3. Minimal Validation: As a pure subtractive pass, it maintains IR integrity 
 *    by definition, allowing it to bypass expensive post-pass validation.
 */

class DeadCodeEliminationPass final : public IPass {
public:
    DeadCodeEliminationPass() = default;

    std::string_view name() const override { return "sc_low.dce"; }

    bool requiresValidation() const override { return false; }

    [[nodiscard]]
    PassResult runOnBlock(sir::Block& block) override;

private:

    bool isLive(const sir::Operation* op) const;
};

} // namespace seecpp::middle
