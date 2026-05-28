#include "source/middle_end/pass_manager.h"

#include <chrono>
#include <format>
#include <iostream>

#include "include/utility/logger.hpp"

namespace seecpp::middle_end {

void PassManager::AddPass(std::unique_ptr<Pass> pass) {
  if (pass != nullptr) {
    passes_.push_back(std::move(pass));
  }
}

std::expected<bool, PassError> PassManager::Run(sir::Block& block) {
  bool graph_changed_overall = false;

  utility::Logger::info(std::format(
      "PassManager: Initiating pipeline execution ({} registered passes).",
      passes_.size()));

  for (const auto& pass : passes_) {
    const std::string_view pass_name = pass->name();

    // 1. Instrumentation: Start high-resolution timer
    const auto start_time = std::chrono::high_resolution_clock::now();

    // 2. Execution
    const bool pass_mutated_ir = pass->Run(block);
    graph_changed_overall |= pass_mutated_ir;

    // 3. Instrumentation: Calculate duration
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
            .count();

    // Optimization: If the pass did not mutate the IR, we can safely bypass 
    // verification and IR dumping to save compilation time.
    if (!pass_mutated_ir) {
      utility::Logger::debug(std::format(
          "[PassManager] Skipped '{}' (0 mutations, {} µs)", 
          pass_name, duration_us));
      continue;
    }

    utility::Logger::info(std::format(
        "[PassManager] Applied '{}' ({} µs)", pass_name, duration_us));

    // 4. Debugging: Conditional IR Dumping
    if (context_.print_ir_after_all) {
      std::cout << "\n=== SIR After " << pass_name << " ===\n";
      block.dump(std::cout);
      std::cout << "======================================\n";
    }

    // 5. Verification: Enforce structural invariants
    if (context_.verify_each) {
      if (!block.Verify(context_.diags)) {
        if (context_.diags) {
          context_.diags->Report(diagnostics::Level::Fatal)
              << "Verification failed after pass: " << pass_name;
        }
        // C++20: Just return the error enum. The Result(E) constructor handles it.
        return PassError::kVerificationFailed; 
      }
    }
  }

  utility::Logger::info("PassManager: Pipeline execution completed successfully.");
  return graph_changed_overall;
}

}  // namespace seecpp::middle_end
