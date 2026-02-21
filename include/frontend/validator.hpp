#pragma once
#include "source/middle_end/sir.hpp"
#include <set>
#include <string>

namespace seecpp::frontend {

class Validator {
public:
    // The main entry point for validation
    bool validate(const middle_end::Block& block);

private:
    // Check if we support this specific operator name
    bool isOpSupported(const std::string& mnemonic);
    
    // Check for cycles in the graph (DFS)
    bool hasCycles(const middle_end::Block& block);

    // Set of officially supported "sc_high" operations
    const std::set<std::string> supported_ops = {
        "sc_high.MatMul", "sc_high.Conv", "sc_high.Relu", 
        "sc_high.Add", "sc_high.Constant"
    };
};

} // namespace seecpp::frontend