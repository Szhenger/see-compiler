#pragma once
#include "middle-end/sir.hpp"
#include <vector>
#include <memory>
#include <string>

namespace seecpp::middle_end {

/**
 * @brief Abstract base class for all Graph Transformations.
 * Following MLIR's "Pass" pattern.
 */
class Pass {
public:
    virtual ~Pass() = default;

    // Every pass must provide a unique name for logging and debugging
    virtual std::string getName() const = 0;

    // The core transformation logic
    // Returns true if the pass succeeded, false if it encountered a fatal error
    virtual bool runOnBlock(sir::Block& block) = 0;
};

/**
 * @brief Orchestrates the execution of multiple passes.
 * Ensures the SIR remains valid between transformations.
 */
class PassManager {
public:
    PassManager() = default;

    // Add a pass to the pipeline
    void addPass(std::unique_ptr<Pass> pass) {
        pipeline.push_back(std::move(pass));
    }

    /**
     * @brief Executes the registered passes in order.
     * @return true if all passes completed successfully.
     */
    bool run(sir::Block& block);

private:
    std::vector<std::unique_ptr<Pass>> pipeline;
};

} // namespace seecpp::middle_end