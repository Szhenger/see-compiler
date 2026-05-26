#include "src/frontend/parser/graph_builder.h"

#include <utility>

namespace seecpp::frontend::parser {

GraphBuilder::GraphBuilder()
    : main_block_(std::make_unique<sir::Block>()),
      insertion_point_(main_block_.get()) {}

void GraphBuilder::SetInsertionPointToEnd(sir::Block* block) {
    insertion_point_ = block;
}

std::expected<sir::Value*, ParseError> GraphBuilder::AddInput(
    std::string_view name, sir::DataType dtype, const sir::Shape& shape) {
    
    std::string symbol(name);
    if (symbol_table_.contains(symbol)) {
        return std::unexpected(ParseError{
            ParseErrorCode::kDuplicateSymbol,
            Location{"GraphInput", "Input"},
            "Input symbol '" + symbol + "' already exists."});
    }

    // Inputs always belong to the top-level main block
    sir::Value* arg = main_block_->addArgument(dtype, shape);
    symbol_table_[symbol] = arg;
    return arg;
}

std::expected<sir::Operation*, ParseError> GraphBuilder::AddOperation(
    const Location& loc,
    std::string_view mnemonic,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names) {

    if (!insertion_point_) {
        return std::unexpected(ParseError{
            ParseErrorCode::kInvalidScope, loc,
            "Attempted to insert operation without an active insertion point."});
    }

    auto op = std::make_unique<sir::Operation>(std::string(mnemonic));
    
    for (const auto& in_name : input_names) {
        auto it = symbol_table_.find(in_name);
        if (it == symbol_table_.end()) {
            return std::unexpected(ParseError{
                ParseErrorCode::kUndeclaredSymbol, loc,
                "Operand '" + in_name + "' accessed before definition."});
        }
        op->addOperand(it->second);
    }

    for (const auto& out_name : output_names) {
        if (symbol_table_.contains(out_name)) {
            return std::unexpected(ParseError{
                ParseErrorCode::kDuplicateSymbol, loc,
                "Operation attempts to redefine existing symbol: " + out_name});
        }
        
        sir::Value* res = op->addResult(out_name, sir::DataType::Unknown, 
                                        sir::Shape{});
        symbol_table_[out_name] = res;
    }

    // Append to the specific insertion point, not just the global block
    return insertion_point_->appendOp(std::move(op));
}

void GraphBuilder::SetAttribute(sir::Operation* op, std::string key, 
                                sir::AttributeValue value) {
    if (op) {
        op->setAttribute(std::move(key), std::move(value));
    }
}

std::expected<std::unique_ptr<sir::Block>, ParseError> GraphBuilder::Finalize() {
    if (!main_block_->validate()) {
        return std::unexpected(ParseError{
            ParseErrorCode::kCycleDetected, 
            Location{"GraphFinalize", "Validation"},
            "The constructed graph violates structural SSA dominance rules."});
    }

    return std::move(main_block_);
}

}  // namespace seecpp::frontend::parser
