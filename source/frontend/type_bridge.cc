#include "seecpp/frontend/type_bridge.h"

// Protobuf headers (compiled from external/onnx)
#include <onnx.pb.h>

#include <stdexcept>

namespace seecpp::frontend {

std::optional<sir::DataType> TypeBridge::mapDataType(int onnx_type) {
    // Maps exhaustively against ONNX TensorProto::DataType
    switch (onnx_type) {
        case onnx::TensorProto::FLOAT:      return sir::DataType::F32;
        case onnx::TensorProto::UINT8:      return sir::DataType::U8;
        case onnx::TensorProto::INT8:       return sir::DataType::I8;
        case onnx::TensorProto::UINT16:     return sir::DataType::U16;
        case onnx::TensorProto::INT16:      return sir::DataType::I16;
        case onnx::TensorProto::INT32:      return sir::DataType::I32;
        case onnx::TensorProto::INT64:      return sir::DataType::I64;
        case onnx::TensorProto::STRING:     return sir::DataType::String;
        case onnx::TensorProto::BOOL:       return sir::DataType::Bool;
        case onnx::TensorProto::FLOAT16:    return sir::DataType::F16;
        case onnx::TensorProto::DOUBLE:     return sir::DataType::F64;
        case onnx::TensorProto::UINT32:     return sir::DataType::U32;
        case onnx::TensorProto::UINT64:     return sir::DataType::U64;
        case onnx::TensorProto::BFLOAT16:   return sir::DataType::BF16;
        
        // ONNX 1.14+ Float8 types (useful for modern ML performance engineering)
        case onnx::TensorProto::FLOAT8E4M3FN:   return sir::DataType::F8E4M3FN;
        case onnx::TensorProto::FLOAT8E4M3FNUZ: return sir::DataType::F8E4M3FNUZ;
        case onnx::TensorProto::FLOAT8E5M2:     return sir::DataType::F8E5M2;
        case onnx::TensorProto::FLOAT8E5M2FNUZ: return sir::DataType::F8E5M2FNUZ;

        // Complex types are mapped if SIR supports them, otherwise fallback
        case onnx::TensorProto::COMPLEX64:
        case onnx::TensorProto::COMPLEX128:
        case onnx::TensorProto::UNDEFINED:
        default:
            return std::nullopt;
    }
}

sir::Shape TypeBridge::mapShape(const onnx::TypeProto_Tensor& tensor_type) {
    sir::Shape shape;
    
    // If the tensor has no shape defined at all, it is completely unranked.
    if (!tensor_type.has_shape()) {
        shape.is_unranked = true;
        return shape;
    }

    shape.is_unranked = false;
    for (const auto& dim : tensor_type.shape().dim()) {
        if (dim.has_dim_value()) {
            // Static dimension (e.g., 224, 3, 64)
            shape.dims.push_back(dim.dim_value());
        } else if (dim.has_dim_param()) {
            // Symbolic dimension (e.g., "batch_size", "seq_len")
            // A production IR needs to track this string for shape inference, 
            // but functionally defaults to kDynamic.
            shape.dims.push_back(sir::Shape::kDynamic);
            shape.symbolic_dims.push_back(dim.dim_param());
        } else {
            // Missing dimension entirely
            shape.dims.push_back(sir::Shape::kDynamic);
            shape.symbolic_dims.emplace_back(""); 
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
            
        case onnx::AttributeProto::STRINGS: {
            std::vector<std::string> strings;
            strings.reserve(attr.strings().size());
            for (const auto& str : attr.strings()) {
                strings.push_back(str);
            }
            return strings;
        }

        case onnx::AttributeProto::TENSOR:
            return mapTensor(attr.t());

        case onnx::AttributeProto::TENSORS: {
            // For operations that take a list of tensors as an attribute
            std::vector<sir::AttributeValue> tensors;
            tensors.reserve(attr.tensors().size());
            for (const auto& t : attr.tensors()) {
                if (auto mapped_t = mapTensor(t)) {
                    tensors.push_back(*mapped_t);
                }
            }
            return tensors;
        }

        // Subgraphs (used in Control Flow ops like 'If', 'Loop', 'Scan')
        // In a strict AOT compiler, subgraphs are usually lifted to top-level functions 
        // during an earlier pass or handled by a dedicated ControlFlowIngressor.
        case onnx::AttributeProto::GRAPH:
        case onnx::AttributeProto::GRAPHS:
        case onnx::AttributeProto::SPARSE_TENSOR:
        case onnx::AttributeProto::SPARSE_TENSORS:
            // TODO(Frontend): Implement mapping for Graphs and Sparse Tensors to SIR variants
            return std::nullopt;
            
        default:
            return std::nullopt;
    }
}

std::optional<sir::AttributeValue> TypeBridge::mapTensor(const onnx::TensorProto& tensor) {
    // Maps an embedded ONNX Tensor into a flat std::vector<uint8_t> representation 
    // wrapped in a SIR TensorAttribute struct, decoupling the raw data from the Protobuf arena.
    
    sir::TensorAttribute sir_tensor;
    sir_tensor.dtype = mapDataType(tensor.data_type()).value_or(sir::DataType::Unknown);
    
    for (auto d : tensor.dims()) {
        sir_tensor.shape.dims.push_back(d);
    }

    if (!tensor.raw_data().empty()) {
        // Fast path: Data is stored as raw contiguous bytes
        const auto* raw_bytes = reinterpret_cast<const uint8_t*>(tensor.raw_data().data());
        sir_tensor.data.assign(raw_bytes, raw_bytes + tensor.raw_data().size());
    } else {
        // Slow path: Data is stored in typed repeated fields (older ONNX generation style)
        // A production compiler must handle these to support legacy models.
        if (tensor.float_data_size() > 0) {
            const size_t bytes = tensor.float_data_size() * sizeof(float);
            sir_tensor.data.resize(bytes);
            std::memcpy(sir_tensor.data.data(), tensor.float_data().data(), bytes);
        } else if (tensor.int64_data_size() > 0) {
            const size_t bytes = tensor.int64_data_size() * sizeof(int64_t);
            sir_tensor.data.resize(bytes);
            std::memcpy(sir_tensor.data.data(), tensor.int64_data().data(), bytes);
        } else if (tensor.int32_data_size() > 0) {
            const size_t bytes = tensor.int32_data_size() * sizeof(int32_t);
            sir_tensor.data.resize(bytes);
            std::memcpy(sir_tensor.data.data(), tensor.int32_data().data(), bytes);
        } else if (tensor.double_data_size() > 0) {
            const size_t bytes = tensor.double_data_size() * sizeof(double);
            sir_tensor.data.resize(bytes);
            std::memcpy(sir_tensor.data.data(), tensor.double_data().data(), bytes);
        } else if (tensor.uint64_data_size() > 0) {
            const size_t bytes = tensor.uint64_data_size() * sizeof(uint64_t);
            sir_tensor.data.resize(bytes);
            std::memcpy(sir_tensor.data.data(), tensor.uint64_data().data(), bytes);
        } else {
            return std::nullopt; // Empty or unsupported embedded tensor
        }
    }

    return sir_tensor;
}

} // namespace seecpp::frontend
