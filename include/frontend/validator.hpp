#pragma once
#include "middle-end/sir.hpp"
#include <set>
#include <string>
#include <unordered_set>

namespace seecpp::frontend {

class Validator {
public:
    /**
     * @brief Performs a multi-pass validation of the SIR Block.
     * Checks: Topology, Op-Support, SSA Dominance, and Type Consistency.
     */
    bool validate(const sir::Block& block);

private:
    // 1. Structural: Is the graph a valid DAG?
    bool checkTopologicalOrder(const sir::Block& block);
    
    // 2. Dialect: Does every op follow its own rules (e.g., MatMul has 2 inputs)?
    bool checkOpConstraints(const sir::Operation& op);

    // 3. Connectivity: Are there any null pointers or undefined SSA values?
    bool checkSSALinks(const sir::Block& block);

    const std::unordered_set<std::string> supported_ops = {
        "sc_high.matmul", "sc_high.conv2d", "sc_high.relu", 
        "sc_high.add", "sc_high.constant", "sc_high.input"
    };
};

} // namespace seecpp::frontend