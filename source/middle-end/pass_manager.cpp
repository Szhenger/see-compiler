#include "include/middle-end/pass_manager.hpp"
#include "include/utility/logger.hpp"

namespace seecpp::middle_end {

bool PassManager::run(sir::Block& block) {
    utility::Logger::info("--- Executing Middle-End Pass Pipeline (" + 
                          std::to_string(pipeline.size()) + " passes) ---");

    for (auto& pass : pipeline) {
        utility::Logger::info("Running Pass: " + pass->getName());

        // 1. Execute the transformation
        if (!pass->runOnBlock(block)) {
            utility::Logger::error("Pass '" + pass->getName() + "' failed!");
            return false;
        }

        // 2. High Rigor: Verification
        // In a full MLIR implementation, we would call Validator::validate() 
        // here to ensure the pass didn't break SSA dominance or leave null pointers.
    }

    utility::Logger::info("--- Pass Pipeline Completed Successfully ---");
    return true;
}

} // namespace seecpp::middle_end