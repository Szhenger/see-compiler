#ifndef SEECPP_FRONTEND_TYPE_BRIDGE_H_
#define SEECPP_FRONTEND_TYPE_BRIDGE_H_

#include "seecpp/sir/sir.h"

#include <optional>
#include <string>
#include <vector>

// Forward declarations for ONNX Protobuf types to keep compilation fast
namespace onnx {
class TypeProto_Tensor;
class AttributeProto;
class TensorProto;
class SparseTensorProto;
class GraphProto;
}

namespace seecpp::frontend {

class TypeBridge {
 public:
    TypeBridge() = delete;

    /// @brief Maps an ONNX TensorProto::DataType integer to a SIR DataType.
    /// Returns std::nullopt if the type is unsupported or unknown.
    static std::optional<sir::DataType> mapDataType(int onnx_type);

    /// @brief Extracts and maps a shape from an ONNX TypeProto_Tensor.
    /// Accounts for fully dynamic (unranked) tensors and symbolic dimensions.
    static sir::Shape mapShape(const onnx::TypeProto_Tensor& tensor_type);

    /// @brief Maps an ONNX AttributeProto to a sir::AttributeValue variant.
    /// Handles scalars, arrays, strings, and embedded constant tensors.
    static std::optional<sir::AttributeValue> mapAttribute(const onnx::AttributeProto& attr);

 private:
    /// @brief Internal helper to extract raw binary or typed data from an ONNX TensorProto
    /// used within an attribute (e.g., constant weights passed as attributes).
    static std::optional<sir::AttributeValue> mapTensor(const onnx::TensorProto& tensor);
};

} // namespace seecpp::frontend

#endif // SEECPP_FRONTEND_TYPE_BRIDGE_H_
