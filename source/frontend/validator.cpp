#include "frontend/validator.hpp"
#include "include/utility/logger.hpp"
#include <unordered_set>

namespace seecpp::frontend {

bool Validator::validate(const sir::Block& block) {
    utility::Logger::info("Validating SIR Graph Integrity...");

    if (!checkSSALinks(block)) return false;
    if (!checkTopologicalOrder(block)) return false;

    for (const auto& op : block.operations) {
        if (supported_ops.find(op->mnemonic) == supported_ops.end()) {
            utility::Logger::error("Unsupported Op: " + op->mnemonic);
            return false;
        }
        if (!checkOpConstraints(*op)) return false;
    }

    utility::Logger::info("Validation Passed.");
    return true;
}

bool Validator::checkTopologicalOrder(const sir::Block& block) {
    std::unordered_set<sir::Value*> defined_values;

    for (const auto& op : block.operations) {
        // 1. Check if all operands for this op have been defined yet
        for (auto* operand : op->operands) {
            if (defined_values.find(operand) == defined_values.end()) {
                utility::Logger::error("SSA Violation: Value used before definition in " + op->mnemonic);
                return false;
            }
        }

        // 2. Mark this op's results as "defined"
        for (const auto& res : op->results) {
            defined_values.insert(res.get());
        }
    }
    return true;
}

bool Validator::checkOpConstraints(const sir::Operation& op) {
    if (op.mnemonic == "sc_high.matmul" && op.operands.size() != 2) {
        utility::Logger::error("MatMul must have exactly 2 operands.");
        return false;
    }
    // Add constraints for Conv2D, etc.
    return true;
}

bool Validator::checkSSALinks(const sir::Block& block) {
    for (const auto& op : block.operations) {
        if (op == nullptr) return false;
        for (auto* operand : op->operands) {
            if (operand == nullptr) {
                utility::Logger::error("Null operand found in op " + op->mnemonic);
                return false;
            }
        }
    }
    return true;
}

} // namespace seecpp::frontend