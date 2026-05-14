#pragma once

#include "middle-end/sir.hpp"
#include <string>
#include <string_view>
#include <unordered_map>
#include <functional>
#include <expected>
#include <memory>
#include <set>

namespace onnx { class GraphProto; class NodeProto; }

namespace seecpp::frontend {

enum class IngestErrorCode {
    InvalidModel,
    UnsupportedOp,
    MissingInput,
    TypeMismatch,
    AttributeError,
    InternalError
};

struct IngestError {
    IngestErrorCode code;
    std::string node_name;
    std::string op_type;
    std::string message;
};

struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const {
        return std::hash<std::string_view>{}(sv);
    }
};

class OnnxIngressor {
public:
    using SymbolTable = std::unordered_map<std::string, sir::Value*, StringHash, std::equal_to<>>;

    using NodeHandler = std::function<
        std::expected<sir::Operation*, IngestError>(
            const onnx::NodeProto&, 
            SymbolTable&, 
            sir::Block&,
            OnnxIngressor*)
    >;

    OnnxIngressor() = default;

    [[nodiscard]]
    std::expected<std::unique_ptr<sir::Block>, IngestError>
    ingest(std::string_view model_path);

private:
    void processInputs(const onnx::GraphProto&, SymbolTable&, sir::Block&);
    void processInitializers(const onnx::GraphProto&, SymbolTable&, sir::Block&);

    [[nodiscard]]
    std::expected<void, IngestError>
    processNodes(const onnx::GraphProto&, SymbolTable&, sir::Block&);

    [[nodiscard]]
    static std::optional<sir::DataType> mapDataType(int onnx_type);

    static const std::unordered_map<std::string, NodeHandler, StringHash, std::equal_to<>> kHandlers;

    std::string current_model_path_;
};

} // namespace seecpp::frontend
