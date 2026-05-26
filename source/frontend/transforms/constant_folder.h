#ifndef SEECPP_TRANSFORMS_CONSTANT_FOLDER_H_
#define SEECPP_TRANSFORMS_CONSTANT_FOLDER_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "seecpp/sir/sir.h"

namespace seecpp::transforms {

/// @brief Signature for operation-specific host evaluation functions.
using FolderFn = std::function<std::unique_ptr<sir::TensorAttribute>(
    sir::Operation*, const std::vector<const sir::TensorAttribute*>&)>;

/// @brief A scalable compile-time interpreter that evaluates static subgraphs
/// using a registry of type-safe, host-side math dispatchers.
class ConstantFolder {
 public:
  ConstantFolder();
  ~ConstantFolder() = default;

  ConstantFolder(const ConstantFolder&) = delete;
  ConstantFolder& operator=(const ConstantFolder&) = delete;

  /// @brief Enrolls a host-evaluator for a specific operator mnemonic.
  void RegisterHandler(std::string_view mnemonic, FolderFn handler);

  /// @brief Iterates through a block and folds static subgraphs.
  /// @return True if the graph structure was mutated.
  bool RunOnBlock(sir::Block* block);

 private:
  std::unordered_map<std::string, FolderFn> registry_;

  /// @brief Delegates to the registry to execute an operation on host memory.
  std::unique_ptr<sir::TensorAttribute> EvaluateOp(
      sir::Operation* op,
      const std::vector<const sir::TensorAttribute*>& operands) const;

  bool IsConstantOp(sir::Operation* op) const;
  const sir::TensorAttribute* GetConstantData(sir::Operation* op) const;
};

}  // namespace seecpp::transforms

#endif  // SEECPP_TRANSFORMS_CONSTANT_FOLDER_H_
