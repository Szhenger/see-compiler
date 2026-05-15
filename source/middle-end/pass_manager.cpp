#include "middle-end/pass_manager.hpp"
#include "utility/logger.hpp"

#include <format>
#include <chrono>
#include <algorithm>
#include <ranges>

namespace seecpp::middle {

[[nodiscard]]
PassResult PassManager::run(sir::Block& block) {
    timings_.clear();
    timings_.reserve(pipeline_.size());

    utility::Logger::info(std::format("PassManager: starting pipeline — {} pass(es) registered", 
                                      pipeline_.size()));

    for (const auto& pass : pipeline_) {
        std::string_view pass_name = pass->name();
        utility::Logger::info(std::format("PassManager: running '{}'", pass_name));

        // --- Execute pass with wall-clock timing ---
        const auto t0 = std::chrono::steady_clock::now();
        
        auto result = pass->runOnBlock(block);
        
        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        timings_.push_back({std::string(pass_name), elapsed_ms});

        // --- Propagate pass failure immediately ---
        if (!result) {
            utility::Logger::error(std::format("PassManager: '{}' failed — {}", 
                                               pass_name, result.error().message));
            return std::unexpected(result.error());
        }

        utility::Logger::info(std::format("PassManager: '{}' completed in {:.2f} ms", 
                                          pass_name, elapsed_ms));

        // --- Inter-pass validation (opt-in per pass) ---
        if (pass->requiresValidation()) {
            utility::Logger::info(std::format("PassManager: running inter-pass validation after '{}'", 
                                              pass_name));

            last_report_ = validator_.validate(block);

            if (last_report_.hasErrors()) {
                // Use C++20 ranges to locate the first hard error declaratively
                auto it = std::ranges::find_if(last_report_.diagnostics, 
                                               [](const auto& d) { return d.isError(); });
                                               
                std::string_view first_error = (it != last_report_.diagnostics.end()) 
                                             ? it->message 
                                             : "Unknown validation error";

                utility::Logger::error(std::format("PassManager: inter-pass validation failed after '{}' — {}", 
                                                   pass_name, first_error));

                return std::unexpected(PassError{
                    std::string(pass_name),
                    std::format("Inter-pass validation failed: {}", first_error)
                });
            }

            // Surface any warnings without aborting
            for (const auto& d : last_report_.diagnostics) {
                if (d.isWarning()) {
                    utility::Logger::warn(std::format("PassManager [validation]: {}", d.message));
                }
            }
        }
    }

    utility::Logger::info("PassManager: pipeline completed successfully.");
    printTimings();
    return {};
}

void PassManager::printTimings() const {
    if (timings_.empty()) return;

    double total_ms = 0.0;
    for (const auto& t : timings_) {
        total_ms += t.elapsed_ms;
    }

    // Accumulate the formatted table into a single string for the logger
    std::string report = "\nPassManager timing summary:\n";
    
    // Header row: 40-char left-aligned, 10-char right-aligned, 10-char right-aligned
    report += std::format("  {:<40}{:>10}{:>10}\n", "Pass", "ms", "%");
    report += std::format("  {:-<60}\n", ""); // 60 dashes

    for (const auto& t : timings_) {
        const double pct = (total_ms > 0.0) ? (t.elapsed_ms / total_ms) * 100.0 : 0.0;
        // Data row: {:.2f} enforces 2 decimal places
        report += std::format("  {:<40}{:>10.2f}{:>9.2f}%\n", t.pass_name, t.elapsed_ms, pct);
    }

    report += std::format("  {:-<60}\n", "");
    report += std::format("  {:<40}{:>10.2f}\n", "Total", total_ms);

    utility::Logger::info(report);
}

} // namespace seecpp::middle
