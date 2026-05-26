#include "source/transforms/constant_folder.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "seecpp/sir/sir.h"

namespace seecpp::transforms {

namespace {

// --- Host Math Dispatch Engine ---

size_t GetElementCount(const sir::Shape& shape) {
  if (shape.is_unranked) return 0;
  size_t count = 1;
  for (int64_t dim : shape.dims) {
    if (dim == sir::Shape::kDynamic) return 0;
    count *= dim;
  }
  return count;
}

// Generalized dispatcher for binary element-wise operations
template <typename T, typename OpCode>
void EvaluateBinaryOp(const uint8_t* lhs, const uint8_t* rhs, uint8_t* dst,
                      size_t elements, OpCode op_func) {
  const T* typed_lhs = reinterpret_cast<const T*>(lhs);
  const T* typed_rhs = reinterpret_cast<const T*>(rhs);
  T* typed_dst = reinterpret_cast<T*>(dst);

  for (size_t i = 0; i < elements; ++i) {
    typed_dst[i] = op_func(typed_lhs[i], typed_rhs[i]);
  }
}

}  // namespace

ConstantFolder::ConstantFolder() {
  // Registering Add Operator
  RegisterHandler("sc_high.Add", [](
      sir::Operation* op,
      const std::vector<const sir::TensorAttribute*>& operands) 
      -> std::unique_ptr<sir::TensorAttribute> {
    
    if (operands.size() != 2) return nullptr;

    auto result = std::make_unique<sir::TensorAttribute>();
    result->dtype = op->getResult(0)->getDataType();
    result->shape = op->getResult(0)->getShape();

    size_t count = GetElementCount(result->shape);
    if (count == 0) return nullptr;

    size_t bytes = count * (result->dtype == sir::DataType::F32 ? 4 : 4);
    result->data.resize(bytes);

    if (result->dtype == sir::DataType::F32) {
      EvaluateBinaryOp<float>(operands[0]->data.data(),
                              operands[1]->data.data(),
                              result->data.data(), count, std::plus<float>());
    } else if (result->dtype == sir::DataType::I32) {
      EvaluateBinaryOp<int32_t>(operands[0]->data.data(),
                                operands[1]->data.data(),
                                result->data.data(), count, std::plus<int32_t>());
    } else {
      return nullptr;
    }
    return result;
  });

  // Registering Shape Operator (Metadata folding, no math required)
  RegisterHandler("sc_high.Shape", [](
      sir::Operation* op,
      const std::vector<const sir::TensorAttribute*>& operands) 
      -> std::unique_ptr<sir::TensorAttribute> {
    
    if (operands.size() != 1) return nullptr;

    auto result = std::make_unique<sir::TensorAttribute>();
    result->dtype = sir::DataType::I64; 
    
    const sir::Shape& input_shape = operands[0]->shape;
    result->shape = sir::Shape{false, {static_cast<int64_t>(input_shape.dims.size())}};
    
    // Allocate space and pack the shape dimensions as the tensor payload
    size_t bytes = input_shape.dims.size() * sizeof(int64_t);
    result->data.resize(bytes);
    std::memcpy(result->data.data(), input_shape.dims.data(), bytes);
    
    return result;
  });
}

void ConstantFolder::RegisterHandler(std::string_view mnemonic,
                                     FolderFn handler) {
  registry_[std::string(mnemonic)] = std::move(handler);
}

bool ConstantFolder::IsConstantOp(sir::Operation* op) const {
  return op->getMnemonic() == "sc_high.Constant";
}

const sir::TensorAttribute* ConstantFolder::GetConstantData(
    sir::Operation* op) const {
  auto attr = op->getAttribute("value");
  if (!attr) return nullptr;
  return std::get_if<sir::TensorAttribute>(&(*attr));
}

std::unique_ptr<sir::TensorAttribute> ConstantFolder::EvaluateOp(
    sir::Operation* op,
    const std::vector<const sir::TensorAttribute*>& operands) const {
  
  auto it = registry_.find(op->getMnemonic());
  if (it == registry_.end()) {
    return nullptr;  // Operation is not registered for host evaluation
  }
  return it->second(op, operands);
}

bool ConstantFolder::RunOnBlock(sir::Block* block) {
  bool changed = false;
  auto it = block->begin();

  while (it != block->end()) {
    sir::Operation* op = &(*it);
    ++it;  

    if (IsConstantOp(op)) continue;

    std::vector<const sir::TensorAttribute*> constant_inputs;
    constant_inputs.reserve(op->getNumOperands());
    bool all_inputs_constant = true;

    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      sir::Operation* def_op = op->getOperand(i)->getDefiningOp();
      if (def_op && IsConstantOp(def_op)) {
        constant_inputs.push_back(GetConstantData(def_op));
      } else {
        all_inputs_constant = false;
        break;
      }
    }

    if (!all_inputs_constant || constant_inputs.empty()) continue;

    auto folded_tensor = EvaluateOp(op, constant_inputs);
    if (!folded_tensor) continue;

    auto const_op = std::make_unique<sir::Operation>("sc_high.Constant");
    const_op->addResult(op->getMnemonic() + "_folded", 
                        op->getResult(0)->getDataType(),
                        op->getResult(0)->getShape());
    const_op->setAttribute("value", *folded_tensor);

    sir::Operation* inserted_constant = block->insertBefore(op, std::move(const_op));
    
    op->getResult(0)->replaceAllUsesWith(inserted_constant->getResult(0));
    op->dropAllReferences();
    block->eraseOp(op);

    changed = true;
  }
  return changed;
}

}  // namespace seecpp::transforms
