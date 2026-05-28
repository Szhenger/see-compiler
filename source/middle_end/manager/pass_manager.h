#ifndef SEECPP_MIDDLE_END_PASS_MANAGER_H_
#define SEECPP_MIDDLE_END_PASS_MANAGER_H_

#include <expected>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "seecpp/middle_end/pass_context.h"
#include "seecpp/middle_end/passes/pass.h"
#include "seecpp/sir/sir.h"

namespace seecpp::middle_end {

/// @brief Represents failure states during the compilation pipeline.
enum class PassError {
  kVerificationFailed,
  kPassInternalError,
  kInitializationFailed
};

/// @brief Orchestrates the ordered execution of intermediate representation (IR)
/// transformations, enforcing structural invariants and tracking performance.
class PassManager {
 public:
  /// @brief Constructs a PassManager with an immutable configuration context.
  /// @param context Global configuration, diagnostic routing, and debug flags.
  explicit PassManager(PassContext context) : context_(std::move(context)) {}
  ~PassManager() = default;

  // Enforce strict ownership: PassManagers cannot be copied or moved.
  PassManager(const PassManager&) = delete;
  PassManager& operator=(const PassManager&) = delete;

  /// @brief Transfers ownership of a pre-instantiated pass into the pipeline.
  /// @param pass Unique pointer to the constructed pass.
  void AddPass(std::unique_ptr<Pass> pass);

  /// @brief Ergonomic helper to construct and append a pass in-place.
  /// @tparam T The specific Pass class to instantiate.
  /// @tparam Args Argument types forwarded to T's constructor.
  template <typename T, typename... Args>
  void Add(Args&&... args) {
    static_assert(std::is_base_of_v<Pass, T>,
                  "T must inherit from seecpp::middle_end::Pass");
    AddPass(std::make_unique<T>(std::forward<Args>(args)...));
  }

  /// @brief Executes all registered passes sequentially on the given IR block.
  /// @param block The primary SIR block to transform.
  /// @return True if the block was mutated by any pass, or a PassError if 
  /// structural verification failed during execution.
  [[nodiscard]] std::expected<bool, PassError> Run(sir::Block& block);

 private:
  std::vector<std::unique_ptr<Pass>> passes_;
  PassContext context_;
};

}  // namespace seecpp::middle_end

#endif  // SEECPP_MIDDLE_END_PASS_MANAGER_H_
