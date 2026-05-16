#include "middle-end/dead_code_elimin_pass.hpp"
#include "utility/logger.hpp"

#include <unordered_set>
#include <vector>
#include <format>

namespace seecpp::middle {

std::expected<void, PassError>
DeadCodeEliminationPass::runOnBlock(sir::Block& block) {

    std::unordered_set<sir::Operation*> live_ops;
    std::vector<sir::Operation*> worklist;

    // 1a. Identify roots
    block.walk([&](sir::Operation* op) {
        if (op->isMemoryOp() || op->isControlFlow()) {
            live_ops.insert(op);
            worklist.push_back(op);
        }
    });

    // 1b. Propagate liveness backwards via operand definitions (O(N) traversal)
    while (!worklist.empty()) {
        sir::Operation* current = worklist.back();
        worklist.pop_back();

        for (size_t i = 0; i < current->numOperands(); ++i) {
            sir::Value* operand = current->operand(i);
            
            // If the operand is produced by an operation (not a block argument)
            if (sir::Operation* def_op = operand->definingOp()) {
                // If we haven't marked this defining op as live yet, do so and enqueue it
                if (live_ops.insert(def_op).second) {
                    worklist.push_back(def_op);
                }
            }
        }
    }

    std::vector<sir::Operation*> dead_ops;
    block.walk([&](sir::Operation* op) {
        // If it was never reached by the backward propagation, it is dead code.
        if (!live_ops.count(op)) {
            dead_ops.push_back(op);
        }
    });

    if (dead_ops.empty()) {
        return {}; // Early exit, no mutations needed
    }

    for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it) {
        block.removeOp(*it);
    }

    utility::Logger::info(std::format(
        "DeadCodeEliminationPass: removed {} dead op(s)", 
        dead_ops.size()
    ));

    return {};
}

} // namespace seecpp::middle
