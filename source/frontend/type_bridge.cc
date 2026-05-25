#include "seecpp/frontend/type_bridge.h"

// Protobuf headers (compiled from external/onnx)
#include <onnx.pb.h>

namespace seecpp::frontend {

std::optional<sir::DataType> TypeBridge::mapDataType(int onnx_type) {
    switch (onnx_type) {
        case onnx::TensorProto::FLOAT:    return sir::DataType::F32;
        case onnx::TensorProto::INT32:    return sir::DataType::I32;
        case onnx::TensorProto::INT64:    return sir::DataType::I64;
        case onnx::TensorProto::FLOAT16:  return sir::DataType::F16;
        case onnx::TensorProto::DOUBLE:   return sir::DataType::F64;
        case onnx::TensorProto::BFLOAT16: return sir::DataType::BF16;
        case onnx::TensorProto::INT8:     return sir::DataType::I8;
        case onnx::TensorProto::BOOL:     return sir::DataType::Bool;
        
        // We can add UINT8, INT16, etc., as we expand supported kernels in the backend
        default: return std::nullopt;
    }
}

sir::Shape TypeBridge::mapShape(const onnx::TypeProto_Tensor& tensor_type) {
    sir::Shape shape;
    
    if (tensor_type.has_shape()) {
        for (const auto& dim : tensor_type.shape().dim()) {
            if (dim.has_dim_value()) {
                shape.dims.push_back(dim.dim_value());
            } else {
                // If a dimension relies on a param string (e.g., "batch_size") 
                // or is missing entirely, we treat it as dynamic.
                shape.dims.push_back(sir::Shape::kDynamic);
            }
        }
    }
    
    return shape;
}

std::optional<sir::AttributeValue> TypeBridge::mapAttribute(const onnx::AttributeProto& attr) {
    switch (attr.type()) {
        case onnx::AttributeProto::FLOAT:
            return attr.f();
            
        case onnx::AttributeProto::INT:
            return attr.i();
            
        case onnx::AttributeProto::STRING:
            return attr.s();
            
        case onnx::AttributeProto::FLOATS:
            return std::vector<float>(attr.floats().begin(), attr.floats().end());
            
        case onnx::AttributeProto::INTS:
            return std::vector<int64_t>(attr.ints().begin(), attr.ints().end());
            
        // Complex attributes like TENSOR, GRAPH, and SPARSE_TENSOR 
        // are omitted for now as they require deeper structural mapping.
        default:
            return std::nullopt;
    }
}

} // namespace seecpp::frontend
