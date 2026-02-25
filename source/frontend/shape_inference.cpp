#include "include/frontend/shape_inference.hpp"
#include <cmath>

namespace seecpp::frontend {

bool ShapeInferenceEngine::infer(sir::Block& block) {
    utility::Logger::info("Starting Shape Inference Pass...");

    for (auto& op : block.operations) {
        bool success = false;
        
        if (op->mnemonic == "sc_high.MatMul") {
            success = inferMatMul(op.get());
        } else if (op->mnemonic == "sc_high.Conv") {
            success = inferConv2D(op.get());
        } else if (op->mnemonic == "sc_high.Relu" || op->mnemonic == "sc_high.Add") {
            success = inferElementwise(op.get());
        } else if (op->mnemonic == "sc_high.constant" || op->mnemonic == "sc_high.input") {
            // Shapes are already provided by Ingressor for these
            success = true;
        }

        if (!success) {
            utility::Logger::error("Shape inference failed at: " + op->mnemonic);
            return false;
        }
    }

    utility::Logger::info("Shape Inference Pass complete.");
    return true;
}

bool ShapeInferenceEngine::inferMatMul(sir::Operation* op) {
    if (op->operands.size() != 2) return false;

    auto& shapeA = op->operands[0]->shape.dims;
    auto& shapeB = op->operands[1]->shape.dims;

    // Rigor: Check rank and inner-dimension compatibility
    if (shapeA.size() < 2 || shapeB.size() < 2) {
        utility::Logger::error("MatMul requires tensors of rank >= 2");
        return false;
    }

    // [M, K] * [K, N] -> K must match
    if (shapeA.back() != shapeB[shapeB.size() - 2]) {
        utility::Logger::error("Incompatible inner dimensions for MatMul");
        return false;
    }

    // Output is [M, N]
    op->results[0]->shape.dims = {shapeA[0], shapeB.back()};
    return true;
}

bool ShapeInferenceEngine::inferConv2D(sir::Operation* op) {
    // Formula: Output = ((Input + 2*Pad - Dilation*(Kernel-1) - 1) / Stride) + 1
    auto& inputShape = op->operands[0]->shape.dims;  // [N, C, H, W]
    auto& weightShape = op->operands[1]->shape.dims; // [M, C, KH, KW]

    // Rigor: Retrieve attributes safely
    auto strides = std::get<std::vector<int64_t>>(op->attributes["strides"]);
    
    // We calculate H and W
    int64_t out_h = ((inputShape[2] - weightShape[2]) / strides[0]) + 1;
    int64_t out_w = ((inputShape[3] - weightShape[3]) / strides[1]) + 1;

    // Output shape: [Batch, Filters, Out_H, Out_W]
    op->results[0]->shape.dims = {inputShape[0], weightShape[0], out_h, out_w};
    return true;
}

} // namespace seecpp::frontend