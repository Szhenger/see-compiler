#include "seecpp/frontend/shape_inference.h"

#include <algorithm>
#include <string>
#include <utility>

#include "seecpp/sir/sir.h"

namespace seecpp::frontend {

ShapeInferenceEngine::ShapeInferenceEngine() {
  // In a full production pipeline (like LLVM/MLIR), these handlers are often
  // auto-generated via TableGen. Here, we register the core operations manually.

  RegisterHandler("Add", [](sir::Operation* op, const Location& loc)
                  -> std::expected<void, InferenceError> {
    if (op->getNumOperands() != 2) {
      return std::unexpected(InferenceError{
          InferenceErrorCode::kInvalidOperand, loc,
          "Add operation requires exactly 2 operands."});
    }

    const sir::Shape& lhs = op->getOperand(0)->getShape();
    const sir::Shape& rhs = op->getOperand(1)->getShape();

    auto broadcast_result = BroadcastShapes(lhs, rhs, loc);
    if (!broadcast_result) {
      return std::unexpected(broadcast_result.error());
    }

    // Set output shape. Validates type consistency implicitly.
    sir::DataType dtype = op->getOperand(0)->getDataType();
    op->getResult(0)->setShape(broadcast_result.value());
    op->getResult(0)->setDataType(dtype);

    return {};
  });

  RegisterHandler("Relu", [](sir::Operation* op, const Location& loc)
                  -> std::expected<void, InferenceError> {
    if (op->getNumOperands() != 1) {
      return std::unexpected(InferenceError{
          InferenceErrorCode::kInvalidOperand, loc,
          "Relu operation requires exactly 1 operand."});
    }
    
    // Relu is purely element-wise; shape and type propagate 1:1
    op->getResult(0)->setShape(op->getOperand(0)->getShape());
    op->getResult(0)->setDataType(op->getOperand(0)->getDataType());
    return {};
  });
}

void ShapeInferenceEngine::RegisterHandler(std::string_view mnemonic,
                                           ShapeInferenceFn handler) {
  registry_[std::string(mnemonic)] = std::move(handler);
}

std::expected<void, InferenceError> ShapeInferenceEngine::InferShape(
    sir::Operation* op, const Location& loc) const {
  auto it = registry_.find(op->getMnemonic());
  if (it == registry_.end()) {
    return std::unexpected(InferenceError{
        InferenceErrorCode::kUnsupportedOp, loc,
        "No shape inference handler registered for dialect op: " + 
            op->getMnemonic()});
  }
  return it->second(op, loc);
}

std::expected<void, InferenceError> ShapeInferenceEngine::RunOnBlock(
    sir::Block* block) const {
  // A linear walk through an SSA Block guarantees topological traversal
  for (sir::Operation& op : *block) {
    Location loc{op.getMnemonic(), "BlockTraversal"};
    auto result = InferShape(&op, loc);
    if (!result) {
      return result;
    }
  }
  return {};
}

std::expected<sir::Shape, InferenceError> BroadcastShapes(
    const sir::Shape& lhs, const sir::Shape& rhs, const Location& loc) {
  
  if (lhs.is_unranked || rhs.is_unranked) {
    sir::Shape result;
    result.is_unranked = true;
    return result;
  }

  sir::Shape result;
  result.is_unranked = false;
  
  int lhs_rank = lhs.dims.size();
  int rhs_rank = rhs.dims.size();
  int max_rank = std::max(lhs_rank, rhs_rank);
  
  result.dims.resize(max_rank);

  // ONNX broadcasting aligns dimensions from right to left
  for (int i = 0; i < max_rank; ++i) {
    int64_t l_dim = (i < lhs_rank) ? lhs.dims[lhs_rank - 1 - i] : 1;
    int64_t r_dim = (i < rhs_rank) ? rhs.dims[rhs_rank - 1 - i] : 1;

    if (l_dim == r_dim) {
      result.dims[max_rank - 1 - i] = l_dim;
    } else if (l_dim == 1) {
      result.dims[max_rank - 1 - i] = r_dim;
    } else if (r_dim == 1) {
      result.dims[max_rank - 1 - i] = l_dim;
    } else if (l_dim == sir::Shape::kDynamic || 
               r_dim == sir::Shape::kDynamic) {
      // Dynamic dimension overrides static dimensions (unless the static is 1)
      result.dims[max_rank - 1 - i] = sir::Shape::kDynamic;
    } else {
      return std::unexpected(InferenceError{
          InferenceErrorCode::kDimMismatch, loc,
          "Incompatible broadcast dimensions: " + std::to_string(l_dim) + 
          " and " + std::to_string(r_dim)});
    }
  }
  return result;
}

}  // namespace seecpp::frontend
