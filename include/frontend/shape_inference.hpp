#pragma once
#include "middle-end/sir.hpp"
#include "utility/logger.hpp"

namespace seecpp::frontend {

class ShapeInferenceEngine {
    public:
        /**
         * @brief Performs a forward pass over the block to calculate all tensor shapes.
         * @return true if all shapes were successfully resolved.
         */
        bool infer(sir::Block& block);

    private:
        // --- The Core Math Logic ---
        
        // Handles Matrix Multiplication: [M, K] x [K, N] -> [M, N]
        bool inferMatMul(sir::Operation* op);

        // Handles Convolution: Calculates output based on padding, stride, and dilation
        bool inferConv2D(sir::Operation* op);

        // Handles Add/Sub/Mul: Implements NumPy-style broadcasting rules
        bool inferElementwise(sir::Operation* op);

        // --- Helper for Rigor ---
        
        // Ensures all operands of an operation actually have a shape before we use them
        bool verifyOperandsDefined(sir::Operation* op);
};

} // namespace seecpp::frontend