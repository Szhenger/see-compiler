#ifndef SEECPP_FRONTEND_VALIDATOR_H_
#define SEECPP_FRONTEND_VALIDATOR_H_

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "seecpp/sir/sir.h"

namespace seecpp::frontend {

/// @brief Represents a single structural or semantic violation in the IR.
struct ValidationError {
  enum class Severity { Warning, Error };
  Severity severity;
  std::string op_mnemonic;
  std::string value_id;
  std::string message;
};

/// @brief Aggregates the results of a validation pass.
struct ValidationReport {
  std::vector<ValidationError> diagnostics;

  [[nodiscard]] bool HasErrors() const {
    for (const auto& diag : diagnostics) {
      if (diag.severity == ValidationError::Severity::Error) {
        return true;
      }
    }
    return false;
  }
};

/// @brief Enforces structural integrity and semantic invariants across an SIR
/// block, guaranteeing safety for subsequent transformations and lowering.
class Validator {
 public:
  Validator() = default;
  ~Validator() = default;

  /// @brief Executes a multi-pass validation over the provided block.
  [[nodiscard]] ValidationReport Validate(const sir::Block& block) const;

 private:
  static const std::unordered_set<std::string> kSupportedOps;

  void CheckSsaLinks(const sir::Block& block, ValidationReport& report) const;
  
  void CheckTopologicalOrder(const sir::Block& block, 
                             ValidationReport& report) const;
                             
  void CheckOpConstraints(const sir::Operation& op, 
                          ValidationReport& report) const;
                          
  void CheckTypeConsistency(const sir::Operation& op, 
                            ValidationReport& report) const;
                            
  void CheckShapeConsistency(const sir::Operation& op, 
                             ValidationReport& report) const;
};

}  // namespace seecpp::frontend

#endif  // SEECPP_FRONTEND_VALIDATOR_H_
