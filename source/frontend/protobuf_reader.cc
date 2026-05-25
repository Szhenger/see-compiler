#include "seecpp/frontend/protobuf_reader.h"

// Note: Ensure these headers exist in your include/utility/ directory
#include "seecpp/utility/logger.h"
#include "seecpp/utility/weight_buffer.h"

// Protobuf headers (compiled from external/onnx)
#include <onnx.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <fstream>
#include <set>

namespace seecpp::frontend {

// Assuming this is defined and allocated in your utility module
extern utility::WeightBuffer global_weight_buffer;

const std::unordered_map<std::string, ProtobufReader::NodeHandler, StringHash, std::equal_to<>>
ProtobufReader::kHandlers = {

    {"Conv", [](const onnx::NodeProto& node, SymbolTable& sym, sir::Block& block, ProtobufReader* ingressor) 
        -> std::expected<sir::Operation*, IngestError> {

        if (node.input_size() < 2) {
            return std::unexpected(IngestError{IngestErrorCode::MissingInput, node.name(), node.op_type(), "Missing X or W"});
        }

        auto it_in = sym.find(node.input(0));
        auto it_w = sym.find(node.input(1));

        if (it_in == sym.end() || it_w == sym.end()) {
            return std::unexpected(IngestError{IngestErrorCode::MissingInput, node.name(), node.op_type(), "Inputs not in symbol table"});
        }

        sir::Value* input = it_in->second;
        sir::Value* filter = it_w->second;
        sir::Value* bias = nullptr;

        if (node.input_size() > 2) {
            if (auto it_b = sym.find(node.input(2)); it_b != sym.end()) {
                bias = it_b->second;
            }
        }

        std::vector<int64_t> strides = {1, 1}, pads = {0, 0, 0, 0}, dilations = {1, 1};
        int64_t group = 1;

        for (const auto& attr : node.attribute()) {
            if (attr.name() == "strides") strides.assign(attr.ints().begin(), attr.ints().end());
            else if (attr.name() == "pads") pads.assign(attr.ints().begin(), attr.ints().end());
            else if (attr.name() == "dilations") dilations.assign(attr.ints().begin(), attr.ints().end());
            else if (attr.name() == "group") group = attr.i();
        }

        auto* op = block.appendOp("sc_high.conv2d");
        op->addOperand(input);
        op->addOperand(filter);
        if (bias) op->addOperand(bias);

        op->setAttribute("strides", strides);
        op->setAttribute("pads", pads);
        op->setAttribute("dilations", dilations);
        op->setAttribute("group", group);

        // Uses default empty shape until ShapeInferencePass resolves it
        op->addResult(node.output(0), sir::DataType::F32, sir::Shape{});
        return op;
    }},

    {"Gemm", [](const onnx::NodeProto& node, SymbolTable& sym, sir::Block& block, ProtobufReader* ingressor) 
        -> std::expected<sir::Operation*, IngestError> {

        if (node.input_size() < 2) {
            return std::unexpected(IngestError{IngestErrorCode::MissingInput, node.name(), node.op_type(), "Missing A or B"});
        }

        auto it_a = sym.find(node.input(0));
        auto it_b = sym.find(node.input(1));

        if (it_a == sym.end() || it_b == sym.end()) {
            return std::unexpected(IngestError{IngestErrorCode::MissingInput, node.name(), node.op_type(), "Operands not in symbol table"});
        }

        bool trans_a = false, trans_b = false;
        float alpha = 1.0f, beta = 1.0f;

        for (const auto& attr : node.attribute()) {
            if (attr.name() == "transA") trans_a = attr.i() != 0;
            else if (attr.name() == "transB") trans_b = attr.i() != 0;
            else if (attr.name() == "alpha") alpha = attr.f();
            else if (attr.name() == "beta") beta = attr.f();
        }

        auto* op = block.appendOp("sc_high.gemm");
        op->addOperand(it_a->second);
        op->addOperand(it_b->second);
        if (node.input_size() > 2) {
            if (auto it_c = sym.find(node.input(2)); it_c != sym.end()) {
                op->addOperand(it_c->second);
            }
        }

        op->setAttribute("transA", static_cast<int64_t>(trans_a));
        op->setAttribute("transB", static_cast<int64_t>(trans_b));
        op->setAttribute("alpha", alpha);
        op->setAttribute("beta", beta);

        op->addResult(node.output(0), sir::DataType::F32, sir::Shape{});
        return op;
    }},

    {"Relu", [](const onnx::NodeProto& node, SymbolTable& sym, sir::Block& block, ProtobufReader* ingressor) 
        -> std::expected<sir::Operation*, IngestError> {
        
        auto it = sym.find(node.input(0));
        if (it == sym.end()) {
            return std::unexpected(IngestError{IngestErrorCode::MissingInput, node.name(), node.op_type(), "Input missing"});
        }

        auto* op = block.appendOp("sc_high.relu");
        op->addOperand(it->second);
        op->addResult(node.output(0), sir::DataType::F32, sir::Shape{});
        return op;
    }}
};

std::expected<std::unique_ptr<sir::Block>, IngestError>
ProtobufReader::ingest(std::string_view model_path) {
    current_model_path_ = std::string(model_path);
    
    std::ifstream ifs(current_model_path_, std::ios::binary);
    if (!ifs) {
        return std::unexpected(IngestError{IngestErrorCode::InvalidModel, "", "", "Could not open: " + current_model_path_});
    }

    onnx::ModelProto model;
    google::protobuf::io::IstreamInputStream zero_copy_input(&ifs);
    if (!model.ParseFromZeroCopyStream(&zero_copy_input)) {
        return std::unexpected(IngestError{IngestErrorCode::InvalidModel, "", "", "Protobuf parse failed"});
    }

    auto block = std::make_unique<sir::Block>();
    SymbolTable sym;

    processInitializers(model.graph(), sym, *block);
    processInputs(model.graph(), sym, *block);

    auto result = processNodes(model.graph(), sym, *block);
    if (!result) return std::unexpected(result.error());

    return block;
}

void ProtobufReader::processInitializers(const onnx::GraphProto& graph, SymbolTable& sym, sir::Block& block) {
    for (const auto& init : graph.initializer()) {
        auto dt = mapDataType(init.data_type()).value_or(sir::DataType::F32);
        sir::Shape shape;
        for (auto d : init.dims()) shape.dims.push_back(d);

        auto* op = block.appendOp("sc_high.constant");
        op->setAttribute("weight_ref", init.name());
        sym[init.name()] = op->addResult(init.name(), dt, shape);

        if (!init.raw_data().empty()) {
            const float* ptr = reinterpret_cast<const float*>(init.raw_data().data());
            global_weight_buffer.addWeight(init.name(), std::vector<float>(ptr, ptr + shape.volume()));
        } else if (!init.float_data().empty()) {
            global_weight_buffer.addWeight(init.name(), std::vector<float>(init.float_data().begin(), init.float_data().end()));
        }
    }
}

void ProtobufReader::processInputs(const onnx::GraphProto& graph, SymbolTable& sym, sir::Block& block) {
    std::set<std::string_view> initializers;
    for (const auto& init : graph.initializer()) {
        initializers.insert(init.name());
    }

    for (const auto& input : graph.input()) {
        if (initializers.contains(input.name())) continue;

        const auto& tensor_type = input.type().tensor_type();
        auto dt = mapDataType(tensor_type.elem_type()).value_or(sir::DataType::F32);
        
        sir::Shape shape;
        if (tensor_type.has_shape()) {
            for (const auto& dim : tensor_type.shape().dim()) {
                shape.dims.push_back(dim.has_dim_value() ? dim.dim_value() : sir::Shape::kDynamic);
            }
        }

        // The block manages SSA, but we map the ONNX name into our symbol table for later lookup
        sym[input.name()] = block.addArgument(dt, shape);
    }
}

std::expected<void, IngestError>
ProtobufReader::processNodes(const onnx::GraphProto& graph, SymbolTable& sym, sir::Block& block) {
    for (const auto& node : graph.node()) {
        auto it = kHandlers.find(node.op_type());
        if (it == kHandlers.end()) {
            return std::unexpected(IngestError{IngestErrorCode::UnsupportedOp, node.name(), node.op_type(), "No handler for node type"});
        }

        auto result = it->second(node, sym, block, this);
        if (!result) return std::unexpected(result.error());

        sir::Operation* op = *result;
        for (int i = 0; i < node.output_size() && i < static_cast<int>(op->numResults()); ++i) {
            sym[node.output(i)] = op->result(i);
        }
    }
    return {};
}

std::optional<sir::DataType> ProtobufReader::mapDataType(int onnx_type) {
    switch (onnx_type) {
        case 1:  return sir::DataType::F32;
        case 6:  return sir::DataType::I32;
        case 7:  return sir::DataType::I64;
        case 10: return sir::DataType::F16;
        case 11: return sir::DataType::F64;
        case 16: return sir::DataType::BF16;
        default: return std::nullopt;
    }
}

} // namespace seecpp::frontend
