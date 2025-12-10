#ifndef IR_HPP
#define IR_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace ir {

struct Type;
struct Value;
struct Instruction;
struct BasicBlock;
struct Function;
struct Module;

// Small, extensible type system for IR. Types are allocated/owned by Module.
enum class TypeKind : std::uint8_t {
    Void,
    Int,       // integer with width (bits)
    Float,     // double (64-bit) or float (32-bit) by width
    Pointer,
    Array,
    Function
};

struct Type {
    TypeKind kind;
    // metadata
    std::uint32_t width;        // for Int (bits) and Float (bits)
    Type* element_type;         // for Pointer and Array
    std::uint64_t array_length; // for Array
    // for function types:
    Type* return_type;
    std::vector<Type*> param_types;
    bool is_vararg;

    // human readable name (optional)
    std::string name;
};

// A Value represents a typed entity: constant, function, global, local (instruction)
enum class ValueKind : std::uint8_t {
    Undefined,
    ConstantInt,
    ConstantFloat,
    ConstantNull,
    GlobalVariable,
    Argument,
    InstructionValue,
    BasicBlockValue   // optional: blocks as values (for SSA branches)
};

struct Value {
    ValueKind kind;
    std::string name;  // textual name (may be empty for temporaries)
    Type* type;        // pointer to type object (owned by Module)

    // Constant payloads (overloaded fields; interpret by kind)
    std::int64_t const_int;    // for ConstantInt
    double      const_float;   // for ConstantFloat

    // For InstructionValue: pointer to the producing instruction
    Instruction* inst;

    // For GlobalVariable: initializer value (may be nullptr)
    Value* initializer;

    // For Argument: owning Function and index
    Function* parent_fn;
    std::uint32_t arg_index;

    // Debug/source info (optional)
    std::string debug_info;
};

// OpCode: the operation performed by the instruction.
enum class OpCode : std::uint16_t {
    // Arithmetic / Logical
    Add, Sub, Mul, Div, Rem,
    And, Or, Xor, Shl, Shr,

    // Floating arithmetic
    FAdd, FSub, FMul, FDiv,

    // Comparisons (returns i1 or bool-like int)
    ICmpEq, ICmpNe, ICmpLt, ICmpLe, ICmpGt, ICmpGe,
    FCmpEq, FCmpNe, FCmpLt, FCmpLe, FCmpGt, FCmpGe,

    // Memory
    Alloca,      // allocate on stack
    Load,
    Store,
    GetElementPtr, // pointer arithmetic

    // Calls and returns
    Call,
    Ret,

    // Control-flow
    Br,       // unconditional branch (to basicblock)
    CondBr,   // conditional branch (cond, truebb, falsebb)
    Phi,      // phi node for SSA merge

    // Other
    Bitcast,
    Nop
};

struct Instruction {
    OpCode op;

    // operands: pointers to Value (arguments). Ownership is external.
    std::vector<Value*> operands;

    // result value (if the instruction produces one). For void instructions set result = nullptr
    Value* result;

    // parent basic block
    BasicBlock* parent;

    // optional textual annotation
    std::string comment;

    // debug/source info (optional)
    std::string debug_info;
};

// A basic block is a sequence of instructions with a single entry and exit points.
struct BasicBlock {
    std::string name;
    Function* parent;                // owning function
    std::vector<Instruction*> instrs; // instructions in order

    // Predecessor / successor lists (maintained by IR builder)
    std::vector<BasicBlock*> preds;
    std::vector<BasicBlock*> succs;
};

// A function contains parameters, basic blocks, and signature information.
struct Function {
    std::string name;
    Type* function_type;          // should be a TypeKind::Function
    std::vector<Value*> arguments; // argument Value objects
    BasicBlock* entry;             // pointer to entry block (may be nullptr until built)
    std::vector<BasicBlock*> blocks;

    bool is_external;              // if true, no body present (external declaration)
    std::string linkage;           // e.g. "internal", "external"

    // Debug/source info (optional)
    std::string debug_info;
};

// Top-level container for functions, globals, and types. The module is expected to
// "own" the memory for its contents. Provide factory functions in ir.cpp to
// allocate and manage these objects (not included here).
struct Module {
    std::string name;
    std::vector<Type*> types;         // types owned by module
    std::vector<Value*> globals;      // global variables (Value* with kind GlobalVariable)
    std::vector<Function*> functions; // functions owned by module

    // helper maps or metadata may be added in implementation
};

// Type factories
Type* create_void_type(Module* m);
Type* create_int_type(Module* m, std::uint32_t bits);    // e.g., 32 bits
Type* create_float_type(Module* m, std::uint32_t bits);  // 32 or 64
Type* create_pointer_type(Module* m, Type* element);
Type* create_array_type(Module* m, Type* element, std::uint64_t length);
Type* create_function_type(Module* m, Type* ret, const std::vector<Type*>& params, bool is_vararg);

// Value / Constant factories
Value* create_constant_int(Module* m, Type* int_type, std::int64_t value, const std::string& name = "");
Value* create_constant_float(Module* m, Type* float_type, double value, const std::string& name = "");
Value* create_constant_null(Module* m, Type* ty, const std::string& name = "");

Value* create_global_variable(Module* m, Type* ty, const std::string& name, Value* initializer = nullptr, const std::string& linkage = "external");
Value* create_argument(Function* fn, Type* ty, const std::string& name, std::uint32_t index);

// Function / BasicBlock / Instruction factories
Function* create_function(Module* m, Type* fn_type, const std::string& name, bool is_external = false);
BasicBlock* create_basic_block(Function* f, const std::string& name);

Instruction* create_instruction(OpCode op, BasicBlock* parent, const std::vector<Value*>& operands = {}, Value* result = nullptr, const std::string& comment = "");

// Helpers to append instruction to a block and wire operands
Instruction* append_instruction(BasicBlock* bb, Instruction* inst);

// Convenience builders for common instructions:
// arithmetic -> returns Instruction* (and sets result Value* if non-void)
Instruction* build_add(BasicBlock* bb, Value* lhs, Value* rhs, Value* dest = nullptr);
Instruction* build_sub(BasicBlock* bb, Value* lhs, Value* rhs, Value* dest = nullptr);
Instruction* build_mul(BasicBlock* bb, Value* lhs, Value* rhs, Value* dest = nullptr);
Instruction* build_div(BasicBlock* bb, Value* lhs, Value* rhs, Value* dest = nullptr);

// memory
Instruction* build_alloca(BasicBlock* bb, Type* ty, Value* dest = nullptr);
Instruction* build_load(BasicBlock* bb, Value* ptr, Value* dest = nullptr);
Instruction* build_store(BasicBlock* bb, Value* ptr, Value* val);

// control flow
Instruction* build_br(BasicBlock* bb, BasicBlock* target);
Instruction* build_condbr(BasicBlock* bb, Value* cond, BasicBlock* true_bb, BasicBlock* false_bb);
Instruction* build_ret(BasicBlock* bb, Value* val = nullptr);

// phi / call
Instruction* build_phi(BasicBlock* bb, Type* ty, Value* dest = nullptr);
Instruction* build_call(BasicBlock* bb, Function* callee, const std::vector<Value*>& args, Value* dest = nullptr);

// Utility
void add_edge(BasicBlock* from, BasicBlock* to);
void remove_edge(BasicBlock* from, BasicBlock* to);

std::string type_to_string(const Type* t);
std::string value_to_string(const Value* v);
void dump_instruction(const Instruction* inst, std::string& out);
void dump_basic_block(const BasicBlock* bb, std::string& out);
void dump_function(const Function* fn, std::string& out);
void dump_module(const Module* m, std::string& out);

// The Module owns the pointers returned by create_* factories. Provide a
// destroy_module function to release them (implementation in ir.cpp).
void destroy_module(Module* m);

} // namespace ir

#endif // IR_HPP
