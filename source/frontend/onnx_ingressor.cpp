#include "sir.hpp"
#include "onnx.pb.h"
#include <fstream>
#include <iostream>

namespace seecpp::frontend {

class OnnxIngressor {
public:
    // Map to keep track of ONNX string names to our internal SSA Values
    std::map<std::string, sir::Value*> tensor_map;

    bool ingest(const std::string& model_path, sir::Block& block) {
        onnx::ModelProto model;
        std::ifstream in(model_path, std::ios::binary);
        
        if (!model.ParseFromIstream(&in)) {
            std::cerr << "Failed to parse ONNX file." << std::endl;
            return false;
        }

        const auto& graph = model.graph();

        // 1. Process Initializers (Weights) as sc_high.constant ops
        for (const auto& initializer : graph.initializer()) {
            auto op = std::make_unique<sir::Operation>("sc_high.constant");
            op->attributes["name"] = initializer.name();
            
            // Map dimensions
            sir::Shape shape;
            for (auto dim : initializer.dims()) shape.dims.push_back(dim);
            
            auto* val = op->addResult(sir::DataType::F32, shape);
            tensor_map[initializer.name()] = val;
            block.push_back(std::move(op));
        }

        // 2. Process Nodes (Layers)
        for (const auto& node : graph.node()) {
            auto sir_op = std::make_unique<sir::Operation>("sc_high." + node.op_type());

            // Link Operands (Inputs)
            for (const auto& input_name : node.input()) {
                if (tensor_map.count(input_name)) {
                    sir_op->addOperand(tensor_map[input_name]);
                }
            }

            // Extract Attributes (Hyperparameters)
            for (const auto& attr : node.attribute()) {
                if (attr.has_i()) sir_op->attributes[attr.name()] = (int)attr.i();
                else if (attr.ints_size() > 0) {
                    std::vector<int> vals(attr.ints().begin(), attr.ints().end());
                    sir_op->attributes[attr.name()] = vals;
                }
            }

            // Create Results (Outputs)
            for (const auto& output_name : node.output()) {
                // Simplified: assuming F32 for now; real impl uses ONNX shape inference
                auto* val = sir_op->addResult(sir::DataType::F32, {}); 
                tensor_map[output_name] = val;
            }

            block.push_back(std::move(sir_op));
        }
        return true;
    }
};

} // namespace seecpp::frontend