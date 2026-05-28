#ifndef SEECPP_MIDDLE_END_PASS_MANAGER_H_
#define SEECPP_MIDDLE_END_PASS_MANAGER_H_

#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "seecpp/middle_end/pass_context.h"
#include "seecpp/middle_end/passes/pass.h"
#include "seecpp/sir/sir.h"
#include "seecpp/utility/result.h" // <-- C++20 Fallback

namespace seecpp::middle_end {

enum class PassError {
  kVerificationFailed,
  kPassInternalError,
  kInitializationFailed
};

class PassManager {
 public:
  explicit PassManager(PassContext context) : context_(std::move(context)) {}
  ~PassManager() = default;

  PassManager(const PassManager&) = delete;
  PassManager& operator=(const PassManager&) = delete;

  void AddPass(std::unique_ptr<Pass> pass);

  template <typename T, typename... Args>
  void Add(Args&&... args) {
    static_assert(std::is_base_of_v<Pass, T>,
                  "T must inherit from seecpp::middle_end::Pass");
    AddPass(std::make_unique<T>(std::forward<Args>(args)...));
  }

  // C++20 compatible return type
  [[nodiscard]] utility::Result<bool, PassError> Run(sir::Block& block);

 private:
  std::vector<std::unique_ptr<Pass>> passes_;
  PassContext context_;
};

}  // namespace seecpp::middle_end

#endif  // SEECPP_MIDDLE_END_PASS_MANAGER_H_
