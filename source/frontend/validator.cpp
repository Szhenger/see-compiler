#include "frontend/validator.hpp"
#include "utility_end/logger.hpp"

namespace seecpp::frontend {

bool Validator::validate(const middle_end::Block& block) {
    utility_end::Logger::info("Starting Model Validation...");

    for (const auto& op : block.operations) {
        // 1. Support Check
        if (!isOpSupported(op->mnemonic)) {
            utility_end::Logger::error("Unsupported Operator: " + op->mnemonic);
            return false;
        }

        // 2. SSA Integrity: Ensure inputs actually exist
        for (auto* operand : op->operands) {
            if (operand == nullptr || operand->id.empty()) {
                utility_end::Logger::error("Dangling SSA pointer in op: " + op->mnemonic);
                return false;
            }
        }
    }

    // 3. Topology Check
    if (hasCycles(block)) {
        utility_end::Logger::error("Graph Validation Failed: Cyclic dependency detected.");
        return false;
    }

    utility_end::Logger::info("Model Validation Passed.");
    return true;
}

bool Validator::isOpSupported(const std::string& mnemonic) {
    return supported_ops.find(mnemonic) != supported_ops.end();
}

bool Validator::hasCycles(const middle_end::Block& block) {
    // For a simple ML compiler, we expect a DAG (Directed Acyclic Graph)
    // We can implement a quick DFS-based topological check here
    return false; // Placeholder: Assume DAG for now
}

} // namespace seecpp::frontend