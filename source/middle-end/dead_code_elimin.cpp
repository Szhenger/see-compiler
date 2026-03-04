#include "include/middle-end/dead_code_elimin_pass.hpp"
#include "include/utility/logger.hpp"

#include <unordered_set>

namespace seecpp::middle {

std::expected<void, PassError>
DeadCodeEliminationPass::runOnBlock(sir::Block& block) {

    // --- Phase 1: compute the live value set via reverse walk ---
    //
    // A Value is live if:
    //   (a) it is a block argument (live-in by definition), OR
    //   (b) it is a result of a live op.
    //
    // An Operation is live if ANY of its results are consumed (has users),
    // OR if it is a terminator / side-effecting op (sc_mem.*, sc_ctrl.*).
    //
    // We seed liveness from all Values that have at least one user,
    // then propagate backwards through operand chains.

    std::unordered_set<const sir::Value*> live_values;

    // Seed: block arguments are always live.
    for (const auto& arg : block.arguments())
        live_values.insert(arg.get());

    // Seed: any result that has users is live.
    block.walk([&](sir::Operation* op) {
        for (size_t i = 0; i < op->numResults(); ++i) {
            const sir::Value* v = op->result(i);
            if (!v->hasNoUses())
                live_values.insert(v);
        }
    });

    // Backwards propagation: if a result is live, all operands of
    // the defining op are also live (they are consumed to produce it).
    bool changed = true;
    while (changed) {
        changed = false;
        block.walkReverse([&](sir::Operation* op) {
            // Check if this op produces anything live.
            bool op_is_live = false;

            // Side-effecting ops are always live regardless of result use.
            if (op->isMemoryOp() || op->isControlFlow())
                op_is_live = true;

            for (size_t i = 0; i < op->numResults(); ++i) {
                if (live_values.count(op->result(i))) {
                    op_is_live = true;
                    break;
                }
            }

            if (op_is_live) {
                for (size_t i = 0; i < op->numOperands(); ++i) {
                    if (live_values.insert(op->operand(i)).second)
                        changed = true;  // new live value found -> re-sweep
                }
            }
        });
    }

    // --- Phase 2: collect dead ops ---
    std::vector<sir::Operation*> dead_ops;
    block.walk([&](sir::Operation* op) {
        // Op is dead iff all its results are absent from live_values
        // AND it is not side-effecting.
        if (op->isMemoryOp() || op->isControlFlow()) return;

        bool any_result_live = false;
        for (size_t i = 0; i < op->numResults(); ++i) {
            if (live_values.count(op->result(i))) {
                any_result_live = true;
                break;
            }
        }
        if (!any_result_live)
            dead_ops.push_back(op);
    });

    // --- Phase 3: erase dead ops ---
    // Erase in reverse order so that if B depends on A and both are dead,
    // we erase B before A (B must be gone before A's use-def entry is removed).
    for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it)
        block.removeOp(*it);

    utility::Logger::info(
        "DeadCodeEliminationPass: removed " +
        std::to_string(dead_ops.size()) + " dead op(s)");

    return {};
}

} // namespace seecpp::middle