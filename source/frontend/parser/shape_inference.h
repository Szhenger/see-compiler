#ifndef SEECPP_FRONTEND_SHAPE_INFERENCE_H_
#define SEECPP_FRONTEND_SHAPE_INFERENCE_H_

#include <expected>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "seecpp/sir/sir.h"

namespace seecpp::frontend {

/// @brief Represents the provenance of an operation for error reporting.
struct Location {
  std::string node_name;
  std::string op_type;
};

enum class InferenceErrorCode {
  kRankMismatch,
  kDimMismatch,
  kUnsupportedOp,
  kInvalidOperand
};

struct InferenceError {
  InferenceErrorCode code;
  Location loc;
  std::string message;
};

/// @brief Signature for operation-specific shape inference functions.
using ShapeInferenceFn = std::function<std::expected<void, InferenceError>(
    sir::Operation*, const Location&)>;

class ShapeInferenceEngine {
 public:
  ShapeInferenceEngine();
  ~ShapeInferenceEngine() = default;

  ShapeInferenceEngine(const ShapeInferenceEngine&) = delete;
  ShapeInferenceEngine& operator=(const ShapeInferenceEngine&) = delete;

  /// @brief Registers an inference handler for a specific operation mnemonic.
  void RegisterHandler(std::string_view mnemonic, ShapeInferenceFn handler);

  /// @brief Performs shape inference on a single operation via dynamic dispatch.
  std::expected<void, InferenceError> InferShape(
      sir::Operation* op, const Location& loc) const;

  /// @brief Runs shape inference over an entire block in topological order.
  std::expected<void, InferenceError> RunOnBlock(sir::Block* block) const;

 private:
  std::unordered_map<std::string, ShapeInferenceFn> registry_;
};

// --- Production Utility Functions ---

/// @brief Computes the multi-directional broadcasted shape of two inputs.
/// Complies with strict NumPy/ONNX broadcasting semantics.
std::expected<sir::Shape, InferenceError> BroadcastShapes(
    const sir::Shape& lhs, const sir::Shape& rhs, const Location& loc);

}  // namespace seecpp::frontend

#endif  // SEECPP_FRONTEND_SHAPE_INFERENCE_H_
