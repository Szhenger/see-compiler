#include "include/middle-end/sir.hpp"
#include <atomic>

namespace seecpp::sir {

// Internal counter for deterministic SSA naming (e.g., %0, %1, %2)
static std::atomic<size_t> value_id_counter{0};

Value* Operation::addResult(const std::string& custom_id, DataType dt, Shape sh) {
    // 1. Generate a name: Use provided name or auto-increment
    std::string final_id = custom_id.empty() 
        ? "%" + std::to_string(value_id_counter++) 
        : custom_id;

    // 2. Construct the Value (using the refined constructor from hpp)
    auto res = std::make_unique<Value>(final_id, dt, sh, this);
    
    // 3. Move into the results vector and return the raw pointer
    results.push_back(std::move(res));
    return results.back().get();
}

void Operation::addOperand(Value* v) {
    if (!v) return;
    
    operands.push_back(v);
    
    // --- USE-DEF TRACKING ---
    // Automatically register this operation as a 'user' of the value.
    // This is vital for Autodiff: we can now ask a Value "Who used you?"
    v->users.push_back(this);
}

// Factory Method for Conv2D: Standardizing the Ingestion
std::unique_ptr<Operation> createConv2D(Value* input, Value* filter, std::vector<int64_t> strides) {
    auto op = std::make_unique<Operation>("sc_high.conv2d");
    
    // Setting up edges
    op->addOperand(input);
    op->addOperand(filter);
    
    // Storing metadata
    op->attributes["strides"] = std::move(strides);
    
    // Note: Result shape would typically be calculated by the ShapeInferenceEngine later,
    // but we initialize it with a placeholder or empty shape for now.
    op->addResult("", input->dtype, Shape{}); 
    
    return op;
}

} // namespace seecpp::sir