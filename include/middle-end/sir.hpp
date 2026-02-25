#pragma once
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <variant>
#include <optional>

namespace seecpp::sir {

class Operation;

// --- 1. Type System ---
enum class DataType { F32, I32, I64, Bool };

struct Shape {
    std::vector<int64_t> dims;
    bool is_static = true;
    
    // Readability: Helper to get total number of elements
    int64_t volume() const {
        int64_t v = 1;
        for (auto d : dims) v *= d;
        return v;
    }
};

// --- 2. Values and Use-Def tracking ---
struct Value {
    std::string id; 
    DataType dtype;
    Shape shape;
    Operation* defining_op = nullptr; 
    
    // Future-proofing: Track which ops use this value 
    // This makes "Dead Code Elimination" and "In-place updates" possible
    std::vector<Operation*> users; 

    Value(std::string id, DataType dt, Shape sh, Operation* op) 
        : id(std::move(id)), dtype(dt), shape(std::move(sh)), defining_op(op) {}
};

// --- 3. Attributes ---
// Using an alias makes the code less noisy
using AttributeValue = std::variant<int64_t, float, std::string, std::vector<int64_t>>;

// --- 4. The Operation ---
class Operation {
public:
    std::string mnemonic;               
    std::vector<Value*> operands;       
    std::vector<std::unique_ptr<Value>> results; 
    std::map<std::string, AttributeValue> attributes;

    explicit Operation(std::string name) : mnemonic(std::move(name)) {}
    
    // Readability: Fluid interface for building the graph
    void addOperand(Value* v) { 
        operands.push_back(v); 
        // Logic: When we add an operand, this op becomes a "user" of that value
        v->users.push_back(this);
    }

    Value* addResult(const std::string& id, DataType dt, Shape sh) {
        auto val = std::make_unique<Value>(id, dt, sh, this);
        results.push_back(std::move(val));
        return results.back().get();
    }

    // Dialect helper: Is this a high-level op or low-level?
    bool isHighLevel() const { return mnemonic.find("sc_high") == 0; }
    bool isLowLevel() const { return mnemonic.find("sc_low") == 0; }
};

// --- 5. The Block (The Sequence) ---
class Block {
public:
    std::vector<std::unique_ptr<Operation>> operations;
    
    // Metadata: Every block should know if it has been validated
    bool is_validated = false;

    Operation* appendOp(std::string name) {
        auto op = std::make_unique<Operation>(std::move(name));
        operations.push_back(std::move(op));
        return operations.back().get();
    }
    
    // Critical for Day 2: Helper to iterate backwards for Autodiff
    // auto rbegin() { return operations.rbegin(); }
};

} // namespace seecpp::sir