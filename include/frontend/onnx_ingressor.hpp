#pragma once

#include "middle-end/sir.hpp"
#include <string>
#include <string_view>
#include <unordered_map>
#include <functional>
#include <expected>   // C++23 — swap for a custom Status if on C++20

namespace onnx { class GraphProto; class NodeProto; }

namespace seecpp::frontend {

/// Diagnostic returned by ingest(). Carries the failed node name and reason.
struct IngestError {
    std::string node_name;   // ONNX node that triggered the failure
    std::string message;
};

/// Translates an ONNX GraphProto into a seecpp::sir::Block.
///
/// Lifecycle: one ingressor per ingest() call (not reusable).
/// The returned Block is owned by the caller.
class OnnxIngressor {
public:
    OnnxIngressor() = default;

    /// Parse the .onnx file at `model_path` and populate `region` with a
    /// new Block containing the translated SIR operations.
    /// Returns IngestError on any unrecoverable failure.
    [[nodiscard]]
    std::expected<std::unique_ptr<sir::Block>, IngestError>
    ingest(std::string_view model_path);

private:
    // Symbol table: ONNX tensor name -> live SIR Value*.
    // Populated during processInputs / processInitializers,
    // consumed and extended during processNodes.
    using SymbolTable = std::unordered_map<std::string, sir::Value*>;

    // Per-node translation handler signature.
    using NodeHandler = std::function
        std::expected<sir::Operation*, IngestError>(
            const onnx::NodeProto&, SymbolTable&, sir::Block&)
    >;

    void processInputs(const onnx::GraphProto&, SymbolTable&, sir::Block&);
    void processInitializers(const onnx::GraphProto&, SymbolTable&, sir::Block&);

    [[nodiscard]]
    std::expected<void, IngestError>
    processNodes(const onnx::GraphProto&, SymbolTable&, sir::Block&);

    [[nodiscard]]
    static std::optional<sir::DataType> mapDataType(int onnx_type);

    // Dispatch table: populated once at construction.
    static const std::unordered_map<std::string, NodeHandler> kHandlers;

    std::string current_model_path_;   // set by ingest(); used in error messages
};

} // namespace seecpp::frontend