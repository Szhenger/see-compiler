#include "include/frontend/shape_inference.hpp"

namespace seecpp::frontend {

void ShapeInferenceEngine::infer(middle_end::Block& block) {
    for (auto& op : block.operations) {
        if (op->mnemonic == "sc_high.MatMul") {
            inferMatMul(op.get());
        } else if (op->mnemonic == "sc_high.Conv") {
            inferConv2D(op.get());
        }
    }
}

void ShapeInferenceEngine::inferMatMul(middle_end::Operation* op) {
    // SSA logic: MatMul has 2 operands (A and B)
    auto* shapeA = &op->operands[0]->shape;
    auto* shapeB = &op->operands[1]->shape;

    // Basic Stress Test: Check if inner dimensions match
    // Matrix A [M x K] * Matrix B [K x N] -> Result [M x N]
    if (shapeA->dims.back() != shapeB->dims[shapeB->dims.size() - 2]) {
        utility_end::Logger::error("Dimension mismatch in MatMul: " + op->mnemonic);
        return;
    }

    // Assign the inferred shape to the output (Result)
    op->results[0]->shape.dims = {shapeA->dims[0], shapeB->dims.back()};
}

void ShapeInferenceEngine::inferConv2D(middle_end::Operation* op) {
    // Logic for Conv: Out = ((In + 2P - K) / S) + 1
    // This requires reading Attributes like 'strides' and 'pads' 
    // that the OnnxIngressor stored in the Operation.
    auto strides = std::get<std::vector<int>>(op->attributes["strides"]);
}

} // namespace seecpp::frontend