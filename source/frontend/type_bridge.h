#ifndef SEECPP_FRONTEND_TYPE_BRIDGE_H_
#define SEECPP_FRONTEND_TYPE_BRIDGE_H_

#include "seecpp/sir/sir.h"

#include <optional>
#include <string>
#include <vector>

// Forward declarations for ONNX Protobuf types
namespace onnx {
class TypeProto_Tensor;
class AttributeProto;
class TensorProto;
}

namespace seecpp::frontend {

class TypeBridge {
 public:
    // Prevent instantiation; this is a purely static utility class
    TypeBridge() = delete;

    /// @brief Maps an ONNX TensorProto::DataType integer to a SIR DataType.
    static std::optional<sir::DataType> mapDataType(int onnx_type);

    /// @brief Extracts and maps a shape from an ONNX TypeProto_Tensor.
    /// @param tensor_type The ONNX tensor type object.
    /// @return A SIR Shape object. Missing or parameterized dims become kDynamic.
    static sir::Shape mapShape(const onnx::TypeProto_Tensor& tensor_type);

    /// @brief Maps an ONNX AttributeProto to a sir::AttributeValue variant.
    /// @param attr The ONNX attribute object.
    /// @return The mapped AttributeValue, or nullopt if the attribute type is unsupported.
    static std::optional<sir::AttributeValue> mapAttribute(const onnx::AttributeProto& attr);
};

} // namespace seecpp::frontend

#endif // SEECPP_FRONTEND_TYPE_BRIDGE_H_
