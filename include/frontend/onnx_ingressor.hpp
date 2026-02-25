#pragma once // Modern standard
#include "middle-end/sir.hpp"
#include <string>
#include <unordered_map> // Faster lookups

// Forward declarations: SeeC++ doesn't need to 'see' the full ONNX 
// definitions until we are inside the .cpp file.
namespace onnx { class GraphProto; class NodeProto; }

namespace seecpp::frontend {

class OnnxIngressor {
    public:
        OnnxIngressor() = default;

        bool ingest(const std::string& model_path, sir::Block& block);

    private:
        // Renamed for clarity: Symbols can be weights or transient tensors
        std::unordered_map<std::string, sir::Value*> symbol_table;
        
        // Decomposed logic for Rigor:
        void processInputs(const onnx::GraphProto& graph, sir::Block& block);
        void processInitializers(const onnx::GraphProto& graph, sir::Block& block);
        void processNodes(const onnx::GraphProto& graph, sir::Block& block);

        sir::DataType mapDataType(int onnx_type);
};

} // namespace seecpp::frontend