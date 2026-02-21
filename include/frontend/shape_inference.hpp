#pragma once
#include "middle_end/sir.hpp"
#include "utility_end/logger.hpp"

namespace seecpp::frontend {

class ShapeInferenceEngine {
public:
    // The main entry point that iterates through the entire graph
    void infer(middle_end::Block& block);

private:
    // Op-specific inference logic
    void inferMatMul(middle_end::Operation* op);
    void inferConv2D(middle_end::Operation* op);
    void inferElementwise(middle_end::Operation* op); // Relu, Add, etc.
};

} // namespace seecpp::frontend