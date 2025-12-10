#include "ir.hpp"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <sstream>

namespace ir {

// Create a Value and initialize fields. Ownership of values varies:
// - Constants and globals are stored in Module->globals (owned by Module).
// - Instruction temporaries are returned and will be owned by their producing Instruction.
static Value* _alloc_value(ValueKind kind, Type* ty, const std::string& name = "") {
    Value* v = new Value();
    v->kind = kind;
    v->type = ty;
    v->name = name;
    v->const_int = 0;
    v->const_float = 0.0;
    v->inst = nullptr;
    v->initializer = nullptr;
    v->parent_fn = nullptr;
    v->arg_index = 0;
    v->debug_info.clear();
    return v;
}

Type* create_void_type(Module* m) {
    Type* t = new Type();
    t->kind = TypeKind::Void;
    t->width = 0;
    t->element_type = nullptr;
    t->array_length = 0;
    t->return_type = nullptr;
    t->param_types.clear();
    t->is_vararg = false;
    t->name = "void";
    m->types.push_back(t);
    return t;
}

Type* create_int_type(Module* m, std::uint32_t bits) {
    Type* t = new Type();
    t->kind = TypeKind::Int;
    t->width = bits;
    t->element_type = nullptr;
    t->array_length = 0;
    t->return_type = nullptr;
    t->param_types.clear();
    t->is_vararg = false;
    t->name = "i" + std::to_string(bits);
    m->types.push_back(t);
    return t;
}

Type* create_float_type(Module* m, std::uint32_t bits) {
    Type* t = new Type();
    t->kind = TypeKind::Float;
    t->width = bits;
    t->element_type = nullptr;
    t->array_length = 0;
    t->return_type = nullptr;
    t->param_types.clear();
    t->is_vararg = false;
    t->name = (bits == 32 ? "float" : "double");
    m->types.push_back(t);
    return t;
}

Type* create_pointer_type(Module* m, Type* element) {
    Type* t = new Type();
    t->kind = TypeKind::Pointer;
    t->width = 0;
    t->element_type = element;
    t->array_length = 0;
    t->return_type = nullptr;
    t->param_types.clear();
    t->is_vararg = false;
    t->name = element ? (element->name + "*") : std::string("ptr");
    m->types.push_back(t);
    return t;
}

Type* create_array_type(Module* m, Type* element, std::uint64_t length) {
    Type* t = new Type();
    t->kind = TypeKind::Array;
    t->width = 0;
    t->element_type = element;
    t->array_length = length;
    t->return_type = nullptr;
    t->param_types.clear();
    t->is_vararg = false;
    std::ostringstream os;
    os << element->name << "[" << length << "]";
    t->name = os.str();
    m->types.push_back(t);
    return t;
}

Type* create_function_type(Module* m, Type* ret, const std::vector<Type*>& params, bool is_vararg) {
    Type* t = new Type();
    t->kind = TypeKind::Function;
    t->width = 0;
    t->element_type = nullptr;
    t->array_length = 0;
    t->return_type = ret;
    t->param_types = params;
    t->is_vararg = is_vararg;
    std::ostringstream os;
    os << ret->name << " (";
    for (size_t i = 0; i < params.size(); ++i) {
        if (i) os << ", ";
        os << params[i]->name;
    }
    if (is_vararg) {
        if (!params.empty()) os << ", ";
        os << "...";
    }
    os << ")";
    t->name = os.str();
    m->types.push_back(t);
    return t;
}

Value* create_constant_int(Module* m, Type* int_type, std::int64_t value, const std::string& name) {
    Value* v = _alloc_value(ValueKind::ConstantInt, int_type, name);
    v->const_int = value;
    m->globals.push_back(v); // keep in module-owned pool
    return v;
}

Value* create_constant_float(Module* m, Type* float_type, double value, const std::string& name) {
    Value* v = _alloc_value(ValueKind::ConstantFloat, float_type, name);
    v->const_float = value;
    m->globals.push_back(v);
    return v;
}

Value* create_constant_null(Module* m, Type* ty, const std::string& name) {
    Value* v = _alloc_value(ValueKind::ConstantNull, ty, name);
    v->const_int = 0;
    m->globals.push_back(v);
    return v;
}

Value* create_global_variable(Module* m, Type* ty, const std::string& name, Value* initializer, const std::string& linkage) {
    Value* v = _alloc_value(ValueKind::GlobalVariable, ty, name);
    v->initializer = initializer;
    v->debug_info = linkage;
    m->globals.push_back(v);
    return v;
}

Value* create_argument(Function* fn, Type* ty, const std::string& name, std::uint32_t index) {
    Value* v = _alloc_value(ValueKind::Argument, ty, name);
    v->parent_fn = fn;
    v->arg_index = index;
    fn->arguments.push_back(v);
    return v;
}

Function* create_function(Module* m, Type* fn_type, const std::string& name, bool is_external) {
    Function* f = new Function();
    f->name = name;
    f->function_type = fn_type;
    f->arguments.clear();
    f->entry = nullptr;
    f->blocks.clear();
    f->is_external = is_external;
    f->linkage = (is_external ? "external" : "internal");
    f->debug_info.clear();
    m->functions.push_back(f);
    return f;
}

BasicBlock* create_basic_block(Function* f, const std::string& name) {
    BasicBlock* bb = new BasicBlock();
    bb->name = name;
    bb->parent = f;
    bb->instrs.clear();
    bb->preds.clear();
    bb->succs.clear();
    f->blocks.push_back(bb);
    if (!f->entry) f->entry = bb;
    return bb;
}

Instruction* create_instruction(OpCode op, BasicBlock* parent, const std::vector<Value*>& operands, Value* result, const std::string& comment) {
    Instruction* inst = new Instruction();
    inst->op = op;
    inst->operands = operands;
    inst->result = result;
    inst->parent = parent;
    inst->comment = comment;
    inst->debug_info.clear();
    // If result is provided and it's an instruction-value, link back
    if (result) {
        result->inst = inst;
        result->kind = ValueKind::InstructionValue;
    }
    return inst;
}

Instruction* append_instruction(BasicBlock* bb, Instruction* inst) {
    if (!bb || !inst) return inst;
    inst->parent = bb;
    bb->instrs.push_back(inst);
    return inst;
}

static Value* _create_temp_for_inst(Type* ty) {
    // Helper: create a value that will be owned by its producing instruction.
    Value* v = _alloc_value(ValueKind::InstructionValue, ty, std::string()); // nameless temp
    return v;
}

Instruction* build_add(BasicBlock* bb, Value* lhs, Value* rhs, Value* dest) {
    if (!dest) dest = _create_temp_for_inst(lhs ? lhs->type : nullptr);
    Instruction* inst = create_instruction(OpCode::Add, bb, {lhs, rhs}, dest, "add");
    append_instruction(bb, inst);
    // ensure result links back
    if (dest) dest->inst = inst;
    return inst;
}

Instruction* build_sub(BasicBlock* bb, Value* lhs, Value* rhs, Value* dest) {
    if (!dest) dest = _create_temp_for_inst(lhs ? lhs->type : nullptr);
    Instruction* inst = create_instruction(OpCode::Sub, bb, {lhs, rhs}, dest, "sub");
    append_instruction(bb, inst);
    if (dest) dest->inst = inst;
    return inst;
}

Instruction* build_mul(BasicBlock* bb, Value* lhs, Value* rhs, Value* dest) {
    if (!dest) dest = _create_temp_for_inst(lhs ? lhs->type : nullptr);
    Instruction* inst = create_instruction(OpCode::Mul, bb, {lhs, rhs}, dest, "mul");
    append_instruction(bb, inst);
    if (dest) dest->inst = inst;
    return inst;
}

Instruction* build_div(BasicBlock* bb, Value* lhs, Value* rhs, Value* dest) {
    if (!dest) dest = _create_temp_for_inst(lhs ? lhs->type : nullptr);
    Instruction* inst = create_instruction(OpCode::Div, bb, {lhs, rhs}, dest, "div");
    append_instruction(bb, inst);
    if (dest) dest->inst = inst;
    return inst;
}

// memory
Instruction* build_alloca(BasicBlock* bb, Type* ty, Value* dest) {
    if (!dest) dest = _create_temp_for_inst(create_pointer_type(bb->parent->function_type ? bb->parent->function_type->return_type : nullptr, nullptr));
    Instruction* inst = create_instruction(OpCode::Alloca, bb, {}, dest, "alloca");
    append_instruction(bb, inst);
    if (dest) dest->inst = inst;
    return inst;
}

Instruction* build_load(BasicBlock* bb, Value* ptr, Value* dest) {
    if (!dest) dest = _create_temp_for_inst(ptr ? ptr->type ? ptr->type->element_type : nullptr : nullptr);
    Instruction* inst = create_instruction(OpCode::Load, bb, {ptr}, dest, "load");
    append_instruction(bb, inst);
    if (dest) dest->inst = inst;
    return inst;
}

Instruction* build_store(BasicBlock* bb, Value* ptr, Value* val) {
    Instruction* inst = create_instruction(OpCode::Store, bb, {ptr, val}, nullptr, "store");
    append_instruction(bb, inst);
    return inst;
}

// control flow
Instruction* build_br(BasicBlock* bb, BasicBlock* target) {
    // Represent target as a BasicBlockValue (if desired) -- for now store nullptr operand and use comment
    Value* targ_val = nullptr;
    Instruction* inst = create_instruction(OpCode::Br, bb, {}, nullptr, std::string("br ") + target->name);
    append_instruction(bb, inst);
    add_edge(bb, target);
    return inst;
}

Instruction* build_condbr(BasicBlock* bb, Value* cond, BasicBlock* true_bb, BasicBlock* false_bb) {
    Instruction* inst = create_instruction(OpCode::CondBr, bb, {cond}, nullptr, std::string("condbr ") + true_bb->name + " " + false_bb->name);
    append_instruction(bb, inst);
    add_edge(bb, true_bb);
    add_edge(bb, false_bb);
    return inst;
}

Instruction* build_ret(BasicBlock* bb, Value* val) {
    Instruction* inst = create_instruction(OpCode::Ret, bb, val ? std::vector<Value*>{val} : std::vector<Value*>{}, nullptr, "ret");
    append_instruction(bb, inst);
    return inst;
}

// phi / call
Instruction* build_phi(BasicBlock* bb, Type* ty, Value* dest) {
    if (!dest) dest = _create_temp_for_inst(ty);
    Instruction* inst = create_instruction(OpCode::Phi, bb, {}, dest, "phi");
    append_instruction(bb, inst);
    if (dest) dest->inst = inst;
    return inst;
}

Instruction* build_call(BasicBlock* bb, Function* callee, const std::vector<Value*>& args, Value* dest) {
    if (!dest) {
        if (callee && callee->function_type && callee->function_type->return_type && callee->function_type->return_type->kind != TypeKind::Void) {
            dest = _create_temp_for_inst(callee->function_type->return_type);
        } else {
            dest = nullptr;
        }
    }
    // Represent callee as a special operand via comment; pass arguments as operands
    Instruction* inst = create_instruction(OpCode::Call, bb, args, dest, std::string("call ") + callee->name);
    append_instruction(bb, inst);
    if (dest) dest->inst = inst;
    return inst;
}

// Utility
void add_edge(BasicBlock* from, BasicBlock* to) {
    if (!from || !to) return;
    auto fwd = std::find(from->succs.begin(), from->succs.end(), to);
    if (fwd == from->succs.end()) from->succs.push_back(to);
    auto back = std::find(to->preds.begin(), to->preds.end(), from);
    if (back == to->preds.end()) to->preds.push_back(from);
}

void remove_edge(BasicBlock* from, BasicBlock* to) {
    if (!from || !to) return;
    from->succs.erase(std::remove(from->succs.begin(), from->succs.end(), to), from->succs.end());
    to->preds.erase(std::remove(to->preds.begin(), to->preds.end(), from), to->preds.end());
}

std::string type_to_string(const Type* t) {
    if (!t) return "<null-type>";
    std::ostringstream os;
    switch (t->kind) {
        case TypeKind::Void: os << "void"; break;
        case TypeKind::Int: os << "i" << t->width; break;
        case TypeKind::Float: os << (t->width == 32 ? "float" : "double"); break;
        case TypeKind::Pointer:
            os << type_to_string(t->element_type) << "*"; break;
        case TypeKind::Array:
            os << type_to_string(t->element_type) << "[" << t->array_length << "]"; break;
        case TypeKind::Function:
            os << t->return_type ? type_to_string(t->return_type) : "func";
            os << "(";
            for (size_t i = 0; i < t->param_types.size(); ++i) {
                if (i) os << ", ";
                os << type_to_string(t->param_types[i]);
            }
            if (t->is_vararg) { if (!t->param_types.empty()) os << ", "; os << "..."; }
            os << ")";
            break;
        default: os << "<type>"; break;
    }
    return os.str();
}

std::string value_to_string(const Value* v) {
    if (!v) return "<null>";
    std::ostringstream os;
    if (!v->name.empty()) os << v->name;
    else {
        // unnamed value -> print based on kind
        switch (v->kind) {
            case ValueKind::ConstantInt: os << v->const_int; break;
            case ValueKind::ConstantFloat: os << v->const_float; break;
            case ValueKind::ConstantNull: os << "null"; break;
            case ValueKind::Argument: os << "arg" << v->arg_index; break;
            case ValueKind::InstructionValue:
                os << "%tmp" << static_cast<const void*>(v); break;
            case ValueKind::GlobalVariable: os << "@" << v->name; break;
            default: os << "<val>"; break;
        }
    }
    os << ":" << type_to_string(v->type);
    return os.str();
}

void dump_instruction(const Instruction* inst, std::string& out) {
    if (!inst) return;
    std::ostringstream os;
    // result
    if (inst->result) {
        os << value_to_string(inst->result) << " = ";
    } else {
        os << "    ";
    }

    // opcode
    os << "[" << static_cast<int>(inst->op) << "] ";
    // simple mnemonic
    switch (inst->op) {
        case OpCode::Add: os << "add "; break;
        case OpCode::Sub: os << "sub "; break;
        case OpCode::Mul: os << "mul "; break;
        case OpCode::Div: os << "div "; break;
        case OpCode::Rem: os << "rem "; break;
        case OpCode::And: os << "and "; break;
        case OpCode::Or: os << "or "; break;
        case OpCode::Xor: os << "xor "; break;
        case OpCode::Shl: os << "shl "; break;
        case OpCode::Shr: os << "shr "; break;
        case OpCode::FAdd: os << "fadd "; break;
        case OpCode::FSub: os << "fsub "; break;
        case OpCode::FMul: os << "fmul "; break;
        case OpCode::FDiv: os << "fdiv "; break;
        case OpCode::ICmpEq: os << "icmp.eq "; break;
        case OpCode::ICmpNe: os << "icmp.ne "; break;
        case OpCode::ICmpLt: os << "icmp.lt "; break;
        case OpCode::ICmpLe: os << "icmp.le "; break;
        case OpCode::ICmpGt: os << "icmp.gt "; break;
        case OpCode::ICmpGe: os << "icmp.ge "; break;
        case OpCode::Load: os << "load "; break;
        case OpCode::Store: os << "store "; break;
        case OpCode::Alloca: os << "alloca "; break;
        case OpCode::GetElementPtr: os << "gep "; break;
        case OpCode::Call: os << "call "; break;
        case OpCode::Ret: os << "ret "; break;
        case OpCode::Br: os << "br "; break;
        case OpCode::CondBr: os << "condbr "; break;
        case OpCode::Phi: os << "phi "; break;
        case OpCode::Bitcast: os << "bitcast "; break;
        case OpCode::Nop: os << "nop "; break;
        default: os << "op? "; break;
    }

    // operands
    for (size_t i = 0; i < inst->operands.size(); ++i) {
        if (i) os << ", ";
        os << value_to_string(inst->operands[i]);
    }

    if (!inst->comment.empty()) {
        os << " ; " << inst->comment;
    }

    out += os.str();
    out.push_back('\n');
}

void dump_basic_block(const BasicBlock* bb, std::string& out) {
    if (!bb) return;
    std::ostringstream os;
    os << bb->name << ":\n";
    out += os.str();
    for (const Instruction* inst : bb->instrs) {
        dump_instruction(inst, out);
    }
    // successors
    if (!bb->succs.empty()) {
        std::ostringstream s2;
        s2 << "    ; succs:";
        for (const BasicBlock* s : bb->succs) s2 << " " << (s ? s->name : "<null>");
        s2 << "\n";
        out += s2.str();
    }
}

void dump_function(const Function* fn, std::string& out) {
    if (!fn) return;
    std::ostringstream os;
    os << "function " << fn->name << " : " << type_to_string(fn->function_type) << "\n";
    out += os.str();
    if (fn->entry) {
        for (const BasicBlock* bb : fn->blocks) {
            dump_basic_block(bb, out);
        }
    } else {
        out += "  (external)\n";
    }
    out.push_back('\n');
}

void dump_module(const Module* m, std::string& out) {
    if (!m) return;
    std::ostringstream os;
    os << "module " << m->name << "\n";
    out += os.str();
    out += "\n; types\n";
    for (const Type* t : m->types) {
        out += std::string("  ") + type_to_string(t) + "\n";
    }
    out += "\n; globals\n";
    for (const Value* g : m->globals) {
        out += std::string("  ") + value_to_string(g) + "\n";
    }
    out += "\n; functions\n";
    for (const Function* f : m->functions) {
        dump_function(f, out);
    }
}

// -----------------------------
// Memory management: destroy_module
// Walk functions -> blocks -> instructions and free everything.
// -----------------------------
void destroy_module(Module* m) {
    if (!m) return;
    // Delete functions and all nested structures
    for (Function* f : m->functions) {
        if (!f) continue;
        // Delete blocks
        for (BasicBlock* bb : f->blocks) {
            if (!bb) continue;
            // Delete instructions
            for (Instruction* inst : bb->instrs) {
                if (!inst) continue;
                // Delete instruction result if it is an instruction-owned temporary
                if (inst->result && inst->result->kind == ValueKind::InstructionValue && inst->result->inst == inst) {
                    delete inst->result;
                    inst->result = nullptr;
                }
                // Note: operands may point to globals/args/other instruction results; do NOT delete here
                delete inst;
            }
            bb->instrs.clear();
            // Clear pred/succ but do not delete pointers (blocks themselves will be deleted)
            bb->preds.clear();
            bb->succs.clear();
            delete bb;
        }
        f->blocks.clear();
        // delete argument Value objects (they were allocated in create_argument)
        for (Value* arg : f->arguments) {
            if (arg) delete arg;
        }
        f->arguments.clear();
        delete f;
    }
    m->functions.clear();

    // Delete globals and constants
    for (Value* g : m->globals) {
        if (g) delete g;
    }
    m->globals.clear();

    // Delete types
    for (Type* t : m->types) {
        if (t) delete t;
    }
    m->types.clear();

    // Finally, delete the module itself
    delete m;
}

} // namespace ir
