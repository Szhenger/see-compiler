#include "include/middle-end/pass_manager.hpp"

#include <iomanip>
#include <sstream>

namespace seecpp::middle {

[[nodiscard]]
std::expected<void, PassError> PassManager::run(sir::Block& block) {

    timings_.clear();
    timings_.reserve(pipeline_.size());

    utility::Logger::info(
        "PassManager: starting pipeline — " +
        std::to_string(pipeline_.size()) + " pass(es) registered");

    for (const auto& pass : pipeline_) {
        const std::string pass_name(pass->name());
        utility::Logger::info("PassManager: running '" + pass_name + "'");

        // --- Execute pass with wall-clock timing ---
        const auto t0 = std::chrono::steady_clock::now();

        auto result = pass->runOnBlock(block);

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        timings_.push_back({pass_name, elapsed_ms});

        // --- Propagate pass failure immediately ---
        if (!result) {
            utility::Logger::error(
                "PassManager: '" + pass_name + "' failed — " +
                result.error().message);
            return std::unexpected(result.error());
        }

        utility::Logger::info(
            "PassManager: '" + pass_name + "' completed in " +
            [&] {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << elapsed_ms;
                return oss.str();
            }() + " ms");

        // --- Inter-pass validation (opt-in per pass) ---
        if (pass->requiresValidation()) {
            utility::Logger::info(
                "PassManager: running inter-pass validation after '" +
                pass_name + "'");

            last_report_ = validator_.validate(block);

            if (last_report_.hasErrors()) {
                // Collect the first hard error for the PassError message.
                std::string first_error;
                for (const auto& d : last_report_.diagnostics) {
                    if (d.isError()) { first_error = d.message; break; }
                }

                utility::Logger::error(
                    "PassManager: inter-pass validation failed after '" +
                    pass_name + "' — " + first_error);

                return std::unexpected(PassError{
                    pass_name,
                    "Inter-pass validation failed: " + first_error});
            }

            // Surface any warnings without aborting.
            for (const auto& d : last_report_.diagnostics) {
                if (d.isWarning())
                    utility::Logger::warn(
                        "PassManager [validation]: " + d.message);
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
    for (const auto& t : timings_) total_ms += t.elapsed_ms;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "\nPassManager timing summary:\n";
    oss << "  " << std::left << std::setw(40) << "Pass"
        << std::right << std::setw(10) << "ms"
        << std::setw(10) << "%" << "\n";
    oss << "  " << std::string(60, '-') << "\n";

    for (const auto& t : timings_) {
        const double pct = total_ms > 0.0 ? (t.elapsed_ms / total_ms) * 100.0 : 0.0;
        oss << "  " << std::left  << std::setw(40) << t.pass_name
            << std::right << std::setw(10) << t.elapsed_ms
            << std::setw(9)  << pct << "%\n";
    }

    oss << "  " << std::string(60, '-') << "\n";
    oss << "  " << std::left  << std::setw(40) << "Total"
        << std::right << std::setw(10) << total_ms << "\n";

    utility::Logger::info(oss.str());
}

} // namespace seecpp::middle