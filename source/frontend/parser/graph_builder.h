#ifndef SEECPP_FRONTEND_PARSER_GRAPH_BUILDER_H_
#define SEECPP_FRONTEND_PARSER_GRAPH_BUILDER_H_

#include "seecpp/sir/sir.h"

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace seecpp::frontend::parser {

/// @brief Represents the provenance of an operation for error reporting.
struct Location {
    std::string node_name;
    std::string op_type;
    // We can add file/line numbers here later if we ingest text-based IRs.
};

enum class ParseErrorCode {
    kDuplicateSymbol,
    kUndeclaredSymbol,
    kCycleDetected,
    kInvalidEdge,
    kInvalidScope
};

struct ParseError {
    ParseErrorCode code;
    Location loc;
    std::string message;
};

class GraphBuilder {
 public:
    GraphBuilder();
    ~GraphBuilder() = default;

    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;

    /// @brief Changes the active insertion point to support nested subgraphs
    /// (e.g., inside an ONNX 'If' or 'Loop' node).
    void SetInsertionPointToEnd(sir::Block* block);

    /// @brief Registers a graph argument (input tensor) into the global scope.
    std::expected<sir::Value*, ParseError> AddInput(
        std::string_view name, sir::DataType dtype, const sir::Shape& shape);

    /// @brief Orchestrates the addition of a new operation with tracking.
    std::expected<sir::Operation*, ParseError> AddOperation(
        const Location& loc,
        std::string_view mnemonic,
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names);

    /// @brief Safely records an attribute on an explicitly provided operation.
    void SetAttribute(sir::Operation* op, std::string key, 
                      sir::AttributeValue value);

    /// @brief Validates the entire module and returns ownership.
    std::expected<std::unique_ptr<sir::Block>, ParseError> Finalize();

 private:
    // The top-level container for the entire graph
    std::unique_ptr<sir::Block> main_block_;
    
    // The current block where new operations are appended
    sir::Block* insertion_point_;
    
    // Global symbol table tracking string names to SSA values
    std::unordered_map<std::string, sir::Value*> symbol_table_;
};

}  // namespace seecpp::frontend::parser

#endif  // SEECPP_FRONTEND_PARSER_GRAPH_BUILDER_H_
