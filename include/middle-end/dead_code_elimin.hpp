#pragma once

#include "include/middle-end/pass_manager.hpp"
#include "include/middle-end/sir.hpp"

// =============================================================================
// DeadCodeEliminationPass (DCE)
//
// Removes operations whose results are never consumed — "dead" ops.
// This is safe because SIR is in SSA form with full use-def tracking:
//   - An op is dead iff ALL of its result Values have zero users.
//   - Constants whose weight is never read are also dead.
//
// Algorithm: reverse-post-order liveness sweep.
//   1. Walk the block in reverse order.
//   2. An op is live if any of its results appear in the live set.
//   3. On encountering a live op, add all its operands to the live set.
//   4. Any op not marked live is erased.
//
// This pass is cheap (O(n) in op count) and should run after every
// transformation pass that might orphan ops (fusion, constant folding).
//
// Precondition: use-def chains must be accurate (maintained by SIR).
// =============================================================================

namespace seecpp::middle {

class DeadCodeEliminationPass final : public IPass {
public:
    DeadCodeEliminationPass() = default;

    std::string_view name() const override {
        return "DeadCodeEliminationPass";
    }

    // DCE does not restructure the IR — only removes ops.
    // Light enough to skip inter-pass validation for throughput.
    bool requiresValidation() const override { return false; }

    [[nodiscard]]
    std::expected<void, PassError> runOnBlock(sir::Block& block) override;
};

} // namespace seecpp::middle