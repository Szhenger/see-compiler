#include "sir.hpp"

namespace seecpp::sir {

Value* Operation::addResult(DataType dt, Shape sh) {
    auto res = std::make_unique<Value>();
    res->dtype = dt;
    res->shape = sh;
    res->defining_op = this;
    res->id = "%" + std::to_string(reinterpret_cast<uintptr_t>(res.get())); // Basic SSA naming
    
    results.push_back(std::move(res));
    return results.back().get();
}

// Example: Creating a High-Level Dialect Operation (Conv2D)
std::unique_ptr<Operation> createConv2D(Value* input, Value* filter, std::vector<int> strides) {
    auto op = std::make_unique<Operation>("sc_high.conv2d");
    op->addOperand(input);
    op->addOperand(filter);
    op->attributes["strides"] = strides;
    op->addResult(input->dtype, input->shape); 
    return op;
}

} // namespace seecpp::sir