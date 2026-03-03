#pragma once

#include "middle-end/sir.hpp"
#include "frontend/validator.hpp"
#include "utility/logger.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <expected>
#include <chrono>

// =============================================================================
// SeeC++ Middle-End Pass Manager
//
// Architecture (MLIR PassManager-inspired):
//   - Each pass is a self-contained IPass implementation
//   - PassManager owns the pipeline and drives execution
//   - std::expected<void, PassError> threads diagnostics without exceptions
//   - Inter-pass validation is opt-in per pass via requiresValidation()
//   - Timing instrumentation is built-in; no external profiler needed
// =============================================================================

namespace seecpp::middle {

// =============================================================================
// Diagnostics
// =============================================================================

struct PassError {
    std::string pass_name;
    std::string message;
};

// =============================================================================
// IPass — abstract base for all middle-end passes
// =============================================================================

/// Every middle-end transformation implements this interface.
///
/// Naming convention (mirrors MLIR pass registry):
///   sc_high.* -> sc_low.*  : lowering passes    (e.g. "ConvLoweringPass")
///   sc_low.*               : optimisation passes (e.g. "OperatorFusionPass")
///   sc_mem.*               : memory passes       (e.g. "BufferAllocationPass")
class IPass {
public:
    virtual ~IPass() = default;

    /// Human-readable name used in logs and diagnostics.
    virtual std::string_view name() const = 0;

    /// Execute the pass on `block`. Returns PassError on failure.
    [[nodiscard]]
    virtual std::expected<void, PassError> runOnBlock(sir::Block& block) = 0;

    /// If true, the PassManager runs Validator::validate() after this pass
    /// completes and aborts the pipeline if any hard errors are found.
    /// Expensive passes that restructure the IR should return true.
    /// Cheap peephole passes may return false for throughput.
    virtual bool requiresValidation() const { return true; }
};

// =============================================================================
// PassManager
// =============================================================================

/// Owns and drives a linear pipeline of IPass instances over a sir::Block.
///
/// Execution model:
///   1. Each pass runs in registration order.
///   2. If requiresValidation() is true for a pass, Validator::validate()
///      runs immediately after — hard errors abort the pipeline.
///   3. Timing for each pass is recorded and available via timings().
///   4. The full ValidationReport from the last inter-pass validation is
///      accessible via lastReport() for downstream diagnostics.
class PassManager {
public:
    // --- Timing record for one pass ---
    struct PassTiming {
        std::string pass_name;
        double      elapsed_ms = 0.0;
    };

    PassManager()  = default;
    ~PassManager() = default;

    // Non-copyable — owns unique_ptr passes.
    PassManager(const PassManager&)            = delete;
    PassManager& operator=(const PassManager&) = delete;
    PassManager(PassManager&&)                 = default;
    PassManager& operator=(PassManager&&)      = default;

    // --- Pipeline construction ---

    /// Register a pass at the back of the pipeline.
    /// Passes are executed in registration order.
    template <typename PassT, typename... Args>
    PassManager& addPass(Args&&... args) {
        pipeline_.push_back(
            std::make_unique<PassT>(std::forward<Args>(args)...));
        return *this;  // fluent interface: pm.addPass<A>().addPass<B>()
    }

    /// Register an already-constructed pass (for passes with complex setup).
    PassManager& addPass(std::unique_ptr<IPass> pass) {
        pipeline_.push_back(std::move(pass));
        return *this;
    }

    size_t numPasses() const { return pipeline_.size(); }

    // --- Execution ---

    /// Run the full pipeline over `block`.
    /// Returns PassError from the first failing pass (pipeline aborts there).
    [[nodiscard]]
    std::expected<void, PassError> run(sir::Block& block);

    // --- Diagnostics and instrumentation ---

    /// Per-pass wall-clock timings from the most recent run().
    const std::vector<PassTiming>& timings() const { return timings_; }

    /// ValidationReport from the last inter-pass validation that ran.
    /// Empty if no validation pass has executed yet.
    const frontend::ValidationReport& lastReport() const { return last_report_; }

    /// Print a timing summary table to Logger::info.
    void printTimings() const;

private:
    std::vector<std::unique_ptr<IPass>>  pipeline_;
    std::vector<PassTiming>              timings_;
    frontend::ValidationReport           last_report_;
    frontend::Validator                  validator_;
};

} // namespace seecpp::middle