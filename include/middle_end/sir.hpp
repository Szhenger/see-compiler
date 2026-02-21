#include <vector>
#include <string>
#include <memory>
#include <map>
#include <variant>

namespace seecpp::sir {

class Operation;

// 1. The System: Distinguishes between High-Level Tensors and Low-Level MemRefs
enum class DataType { F32, I32, I64 };

struct Shape {
    std::vector<int64_t> dims;
    bool is_static = true;
};

// 2. The Value: Represents an SSA value (the output of an operation)
struct Value {
    std::string id; 
    DataType dtype;
    Shape shape;
    Operation* defining_op = nullptr; // SSA back-pointer
};

// 3. The Operation: The atomic unit of the SIR
class Operation {
public:
    std::string mnemonic;               // e.g., "sc_high.conv2d"
    std::vector<Value*> operands;       // Input SSA values
    std::vector<std::unique_ptr<Value>> results; // Output SSA values
    std::map<std::string, std::variant<int, float, std::string, std::vector<int>>> attributes;

    Operation(std::string name) : mnemonic(std::move(name)) {}
    
    void addOperand(Value* v) { operands.push_back(v); }
    Value* addResult(DataType dt, Shape sh);
};

// 4. The Block: A sequence of operations (The "Function Body")
class Block {
public:
    std::vector<std::unique_ptr<Operation>> operations;
    
    Operation* push_back(std::unique_ptr<Operation> op) {
        operations.push_back(std::move(op));
        return operations.back().get();
    }
};

} // namespace seecpp::sir