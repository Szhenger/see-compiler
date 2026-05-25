#ifndef SEECPP_FRONTEND_PROTOBUF_READER_H_
#define SEECPP_FRONTEND_PROTOBUF_READER_H_

#include "seecpp/sir/sir.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <memory>
#include <expected>
#include <functional>
#include <optional>

// Forward declarations to avoid exposing external/ Protobuf headers to the rest of the project
namespace onnx {
class NodeProto;
class GraphProto;
}

namespace seecpp::frontend {

enum class IngestErrorCode {
    MissingInput,
    InvalidModel,
    UnsupportedOp
};

struct IngestError {
    IngestErrorCode code;
    std::string node_name;
    std::string op_type;
    std::string message;
};

// C++20 Heterogeneous lookup hash for string_view <-> string keys
struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view txt) const {
        return std::hash<std::string_view>{}(txt);
    }
};

using SymbolTable = std::unordered_map<std::string, sir::Value*, StringHash, std::equal_to<>>;

class ProtobufReader {
 public:
    using NodeHandler = std::function<std::expected<sir::Operation*, IngestError>(
        const onnx::NodeProto&, SymbolTable&, sir::Block&, ProtobufReader*)>;

    ProtobufReader() = default;

    std::expected<std::unique_ptr<sir::Block>, IngestError> ingest(std::string_view model_path);

 private:
    void processInitializers(const onnx::GraphProto& graph, SymbolTable& sym, sir::Block& block);
    void processInputs(const onnx::GraphProto& graph, SymbolTable& sym, sir::Block& block);
    std::expected<void, IngestError> processNodes(const onnx::GraphProto& graph, SymbolTable& sym, sir::Block& block);
    
    static std::optional<sir::DataType> mapDataType(int onnx_type);

    std::string current_model_path_;
    static const std::unordered_map<std::string, NodeHandler, StringHash, std::equal_to<>> kHandlers;
};

} // namespace seecpp::frontend

#endif // SEECPP_FRONTEND_PROTOBUF_READER_H_
