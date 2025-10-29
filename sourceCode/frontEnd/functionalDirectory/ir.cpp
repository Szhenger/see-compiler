#include "ir.hpp"

#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <utility>
#include <cstring>   // memcpy
#include <algorithm>

namespace see {

// ========================= Small internal slice copier =========================
template<typename T>
static Slice<T> make_slice(std::vector<T>& storage) {
  return Slice<T>{storage.empty() ? nullptr : storage.data(),
                  static_cast<std::uint32_t>(storage.size())};
}
template<typename T>
static Slice<T> make_slice(const std::vector<T>& storage) {
  return Slice<T>{storage.empty() ? nullptr : storage.data(),
                  static_cast<std::uint32_t>(storage.size())};
}

// ========================= IRBuilder::Impl =========================
struct IRBuilder::Impl {
  explicit Impl(Arch tgt) : target(tgt) {
    // Pre-create void type at id 0
    IRType tv; tv.kind = IRTypeKind::Void;
    types.push_back(tv);
  }

  // -------- Type arena --------
  TypeId push_type(IRType t) {
    types.push_back(t);
    return static_cast<TypeId>(types.size() - 1);
  }

  // -------- Value arena --------
  ValueId push_value(const IRValue& v) {
    values.push_back(v);
    return static_cast<ValueId>(values.size() - 1);
  }

  // Create a “block value” wrapper for encoding block operands where a ValueId is expected
  ValueId make_block_value(FuncId f, BlockId b) {
    IRValue v;
    v.kind = IRValueKind::Inst; // reuse inst ref channel
    v.type = 0;                 // void (unused)
    v.as.inst.func = f;
    v.as.inst.block = b;
    v.as.inst.inst = kInvalidId;
    return push_value(v);
  }

  // -------- Function / block / instruction arenas --------
  struct InstData {
    IRInst inst;                 // public shape
    std::vector<IROperand> ops;  // owning operands (backed for inst.ops slice)
  };
  struct BlockData {
    std::string name;
    std::vector<InstId> insts;
    InstId terminator{kInvalidId};
  };
  struct FuncData {
    std::string      name;
    TypeId           type{kInvalidId};
    std::vector<BlockData> blocks;
    std::vector<ValueId>   params;
    bool             is_decl{true};

    // Mapping from block index to first unfilled phi id list could live here (future)
  };

  // Globals
  struct GlobalData {
    std::string name;
    TypeId      type{kInvalidId};
    bool        is_function{false};
    bool        is_extern{true};
    bool        is_mutable{true};
  };

  // -------- Helpers to access current insertion point --------
  FuncData& cur_func() {
    assert(current_func != kInvalidId && "No current function");
    return funcs.at(current_func);
  }
  BlockData& cur_block() {
    assert(current_func != kInvalidId && current_block != kInvalidId && "No current block");
    return funcs[current_func].blocks.at(current_block);
  }

  // Create instruction in current block
  ValueId emit_inst_value(InstKind k, TypeId ty, const std::vector<ValueId>& operands, std::uint32_t imm0 = 0) {
    auto& B = cur_block();

    InstData data;
    data.inst.kind = k;
    data.inst.type = ty;
    data.ops.reserve(operands.size());
    for (auto v : operands) data.ops.push_back(IROperand{v});
    data.inst.ops = make_slice(data.ops);
    data.inst.imm0 = imm0;

    insts.push_back(std::move(data));
    InstId iid = static_cast<InstId>(insts.size() - 1);
    B.insts.push_back(iid);

    // Non-void instructions yield a ValueId referring back to (func, block, inst)
    if (!is_void_type(type(ty))) {
      IRValue rv;
      rv.kind = IRValueKind::Inst;
      rv.type = ty;
      rv.as.inst = IRInstRef{current_func, current_block, iid};
      return push_value(rv);
    } else {
      return kInvalidId;
    }
  }

  void set_terminator(InstId iid) {
    auto& B = cur_block();
    B.terminator = iid;
  }

  // -------- Public snapshot helpers for finish() --------
  IRType        type(TypeId t) const { return types.at(t); } // copy
  const IRType& type_cref(TypeId t) const { return types.at(t); }
  const IRValue& value_cref(ValueId v) const { return values.at(v); }

  // -------- State --------
  Arch target;

  // Arenas
  std::vector<IRType>   types;   // [0] is Void
  std::vector<IRValue>  values;
  std::vector<FuncData> funcs;
  std::vector<GlobalData> globals;

  std::vector<InstData> insts;   // global inst storage (indexed by InstId)

  // Insertion point
  FuncId  current_func{kInvalidId};
  BlockId current_block{kInvalidId};

  // Persistent buffers to expose slices in IRModule (lifetimes == builder)
  std::vector<IRType>        out_types;
  std::vector<IRValue>       out_values;
  std::vector<IRGlobalObj>   out_globals;
  std::vector<IRFunction>    out_functions;
  std::vector<IRBlock>       out_blocks;
  std::vector<InstId>        out_block_insts;
  std::vector<ValueId>       out_params;
  std::vector<BlockId>       out_fun_blocks;
};

// ========================= IRBuilder API =========================

IRBuilder::IRBuilder(Arch target) : p_(new Impl(target)) {}
// (Optional) manage lifetime with unique_ptr; keep simple for now
// (A production impl would use pimpl with unique_ptr and custom destructor.)
IRModule IRBuilder::finish() {
  // Rebuild public, flat, slice-backed module views based on Impl arenas.

  // Types
  p_->out_types = p_->types;
  // Values
  p_->out_values = p_->values;

  // Globals
  p_->out_globals.clear();
  p_->out_globals.reserve(p_->globals.size());
  for (auto& g : p_->globals) {
    IRGlobalObj G;
    G.name = g.name;
    G.type = g.type;
    G.is_function = g.is_function;
    G.is_extern = g.is_extern;
    G.is_mutable = g.is_mutable;
    p_->out_globals.push_back(G);
  }

  // Functions / Blocks
  p_->out_functions.clear();
  p_->out_blocks.clear();
  p_->out_block_insts.clear();
  p_->out_params.clear();
  p_->out_fun_blocks.clear();

  p_->out_functions.reserve(p_->funcs.size());
  for (std::size_t fi = 0; fi < p_->funcs.size(); ++fi) {
    const auto& F = p_->funcs[fi];

    // Blocks slice indices
    const std::size_t blocks_begin = p_->out_blocks.size();
    for (std::size_t bi = 0; bi < F.blocks.size(); ++bi) {
      const auto& B = F.blocks[bi];
      // Collect insts of this block
      const std::size_t insts_begin = p_->out_block_insts.size();
      p_->out_block_insts.insert(p_->out_block_insts.end(), B.insts.begin(), B.insts.end());
      const std::size_t insts_end = p_->out_block_insts.size();

      IRBlock BB;
      BB.name = B.name;
      BB.insts = Slice<InstId>{
        (insts_begin == insts_end) ? nullptr : &p_->out_block_insts[insts_begin],
        static_cast<std::uint32_t>(insts_end - insts_begin)
      };
      BB.terminator = B.terminator;
      p_->out_blocks.push_back(BB);
    }
    const std::size_t blocks_end = p_->out_blocks.size();

    // Params slice indices
    const std::size_t params_begin = p_->out_params.size();
    p_->out_params.insert(p_->out_params.end(), F.params.begin(), F.params.end());
    const std::size_t params_end = p_->out_params.size();

    IRFunction FF;
    FF.name = F.name;
    FF.type = F.type;
    FF.blocks = Slice<BlockId>{
      (blocks_begin == blocks_end) ? nullptr : reinterpret_cast<BlockId*>(&p_->out_blocks[blocks_begin]),
      static_cast<std::uint32_t>(blocks_end - blocks_begin)
    };
    FF.params = Slice<ValueId>{
      (params_begin == params_end) ? nullptr : &p_->out_params[params_begin],
      static_cast<std::uint32_t>(params_end - params_begin)
    };
    FF.is_decl = F.is_decl;
    p_->out_functions.push_back(FF);
  }

  IRModule M;
  M.target   = p_->target;
  M.types    = make_slice(p_->out_types);
  M.values   = make_slice(p_->out_values);
  M.globals  = make_slice(p_->out_globals);
  M.functions= make_slice(p_->out_functions);
  return M;
}

// -------- Types --------
TypeId IRBuilder::type_void() {
  return 0; // reserved
}
TypeId IRBuilder::type_int(std::uint32_t bit_width) {
  IRType t; t.kind = IRTypeKind::Int; t.bit_width = bit_width;
  return p_->push_type(t);
}
TypeId IRBuilder::type_float(std::uint32_t bit_width) {
  IRType t; t.kind = IRTypeKind::Float; t.bit_width = bit_width;
  return p_->push_type(t);
}
TypeId IRBuilder::type_ptr(TypeId elem, std::uint32_t addrspace) {
  IRType t; t.kind = IRTypeKind::Pointer; t.elem = elem; t.addr_space = addrspace;
  return p_->push_type(t);
}
TypeId IRBuilder::type_func(TypeId ret, const std::vector<TypeId>& params, bool vararg) {
  IRType t; t.kind = IRTypeKind::Function; t.ret = ret; t.is_vararg = vararg;
  // We store params in the type arena by copying them into a tail vector and making a slice into it.
  // For simplicity, we create a dedicated vector per type in a temporary then patch a stable slice on finish().
  // Here we store them directly in the IRType as a Slice pointing to params.data(); to keep it stable,
  // copy into a small owned vector that we keep in Impl::out_types on finish(). Simple approach:
  std::vector<TypeId> temp = params;
  // Temporarily store a slice pointing to this local vector is unsafe, so we instead push the type
  // without params and then patch at finish(). To keep MVP simple, we *do* store the slice
  // as empty here; the front-end can carry param types in the function value’s signature as needed.
  // If you need param slice now, promote to an owned pool. For now:
  t.params = {};
  return p_->push_type(t);
}

// -------- Globals --------
GlobalId IRBuilder::global_var(std::string_view name, TypeId type, bool is_extern, bool is_mutable) {
  IRBuilder::Impl::GlobalData g;
  g.name = std::string(name);
  g.type = type;
  g.is_function = false;
  g.is_extern = is_extern;
  g.is_mutable = is_mutable;
  p_->globals.push_back(std::move(g));
  return static_cast<GlobalId>(p_->globals.size() - 1);
}
GlobalId IRBuilder::global_func(std::string_view name, TypeId func_type, bool is_extern) {
  IRBuilder::Impl::GlobalData g;
  g.name = std::string(name);
  g.type = func_type;
  g.is_function = true;
  g.is_extern = is_extern;
  g.is_mutable = false;
  p_->globals.push_back(std::move(g));
  return static_cast<GlobalId>(p_->globals.size() - 1);
}

// -------- Functions & blocks --------
FuncId IRBuilder::func_begin(std::string_view name, TypeId func_type) {
  IRBuilder::Impl::FuncData f;
  f.name = std::string(name);
  f.type = func_type;
  f.is_decl = true;
  p_->funcs.push_back(std::move(f));
  p_->current_func = static_cast<FuncId>(p_->funcs.size() - 1);
  p_->current_block = kInvalidId;
  return p_->current_func;
}
void IRBuilder::func_set_decl(FuncId f, bool is_decl) {
  p_->funcs.at(f).is_decl = is_decl;
}
ValueId IRBuilder::param_value(FuncId f, std::uint32_t index) {
  auto& F = p_->funcs.at(f);
  // If already created, return existing
  if (index < F.params.size()) return F.params[index];
  // Create on-demand up to index
  while (F.params.size() <= index) {
    // Look up parameter type from function type (we didn't store params in type; assume i64 for MVP)
    // In a full impl, IRType::params carries entries; here we synthesize a placeholder i64.
    TypeId t_i64 = type_int(64);
    IRValue v;
    v.kind = IRValueKind::Param;
    v.type = t_i64;
    v.as.param.func = f;
    v.as.param.type = t_i64;
    v.as.param.index = static_cast<std::uint32_t>(F.params.size());
    F.params.push_back(p_->push_value(v));
  }
  return F.params[index];
}

BlockId IRBuilder::block_create(FuncId f, std::string_view name) {
  auto& F = p_->funcs.at(f);
  IRBuilder::Impl::BlockData b;
  b.name = std::string(name);
  F.blocks.push_back(std::move(b));
  return static_cast<BlockId>(F.blocks.size() - 1);
}
void IRBuilder::block_set_insert_point(FuncId f, BlockId b) {
  p_->current_func = f;
  p_->current_block = b;
}

// -------- Constants --------
ValueId IRBuilder::const_int(TypeId t, std::uint64_t lo, std::uint64_t hi, bool is_signed) {
  IRValue v;
  v.kind = IRValueKind::ConstInt;
  v.type = t;
  v.as.c_int.type = t;
  v.as.c_int.lo = lo;
  v.as.c_int.hi = hi;
  v.as.c_int.is_signed = is_signed;
  return p_->push_value(v);
}
ValueId IRBuilder::const_float_bits(TypeId t, std::uint64_t lo, std::uint64_t hi) {
  IRValue v;
  v.kind = IRValueKind::ConstFloat;
  v.type = t;
  v.as.c_flt.type = t;
  v.as.c_flt.bits_lo = lo;
  v.as.c_flt.bits_hi = hi;
  return p_->push_value(v);
}

// -------- Instructions --------
ValueId IRBuilder::inst_alloca(TypeId obj_type) {
  // result is pointer-to obj_type
  TypeId pty = type_ptr(obj_type);
  return p_->emit_inst_value(InstKind::Alloca, pty, {});
}
ValueId IRBuilder::inst_load(ValueId ptr) {
  // result type should be element type; we cannot infer without type table — assume i64 for MVP
  TypeId t = type_int(64);
  return p_->emit_inst_value(InstKind::Load, t, {ptr});
}
void IRBuilder::inst_store(ValueId value, ValueId ptr) {
  ValueId res = p_->emit_inst_value(InstKind::Store, type_void(), {value, ptr});
  (void)res;
}
ValueId IRBuilder::inst_gep(ValueId base_ptr, const std::vector<ValueId>& indices) {
  // returns a pointer (same elem) — we cannot track pointee in MVP; return ptr type to i8
  TypeId i8 = type_int(8);
  TypeId pty = type_ptr(i8);
  std::vector<ValueId> ops;
  ops.reserve(1 + indices.size());
  ops.push_back(base_ptr);
  ops.insert(ops.end(), indices.begin(), indices.end());
  return p_->emit_inst_value(InstKind::Gep, pty, ops);
}

ValueId IRBuilder::inst_bin(InstKind k, ValueId a, ValueId b) {
  assert((k == InstKind::IAdd || k == InstKind::ISub || k == InstKind::IMul ||
          k == InstKind::UDiv || k == InstKind::SDiv || k == InstKind::URem || k == InstKind::SRem ||
          k == InstKind::Shl || k == InstKind::LShr || k == InstKind::AShr ||
          k == InstKind::And || k == InstKind::Or || k == InstKind::Xor) && "inst_bin: not an int binop");
  // result type = type(a) (MVP)
  TypeId ty = p_->value_cref(a).type;
  return p_->emit_inst_value(k, ty, {a, b});
}
ValueId IRBuilder::inst_fbin(InstKind k, ValueId a, ValueId b) {
  assert((k == InstKind::FAdd || k == InstKind::FSub || k == InstKind::FMul ||
          k == InstKind::FDiv || k == InstKind::FRem) && "inst_fbin: not a float binop");
  TypeId ty = p_->value_cref(a).type;
  return p_->emit_inst_value(k, ty, {a, b});
}

ValueId IRBuilder::inst_icmp(ICmpCond c, ValueId a, ValueId b) {
  // result is i1 (bool)
  TypeId i1 = type_int(1);
  return p_->emit_inst_value(InstKind::ICmp, i1, {a, b}, static_cast<std::uint32_t>(c));
}
ValueId IRBuilder::inst_fcmp(FCmpCond c, ValueId a, ValueId b) {
  TypeId i1 = type_int(1);
  return p_->emit_inst_value(InstKind::FCmp, i1, {a, b}, static_cast<std::uint32_t>(c));
}

ValueId IRBuilder::inst_cast(CastKind c, ValueId v, TypeId to_type) {
  return p_->emit_inst_value(InstKind::Cast, to_type, {v}, static_cast<std::uint32_t>(c));
}

void IRBuilder::inst_br(BlockId target) {
  // create a ValueId that encodes the block
  ValueId bval = p_->make_block_value(p_->current_func, target);
  ValueId res = p_->emit_inst_value(InstKind::Br, type_void(), {bval});
  (void)res;
  p_->set_terminator(static_cast<InstId>(p_->insts.size() - 1));
}
void IRBuilder::inst_condbr(ValueId cond, BlockId then_b, BlockId else_b) {
  ValueId t = p_->make_block_value(p_->current_func, then_b);
  ValueId e = p_->make_block_value(p_->current_func, else_b);
  ValueId res = p_->emit_inst_value(InstKind::CondBr, type_void(), {cond, t, e});
  (void)res;
  p_->set_terminator(static_cast<InstId>(p_->insts.size() - 1));
}
void IRBuilder::inst_ret(ValueId v) {
  ValueId res = p_->emit_inst_value(InstKind::Ret, type_void(), {v});
  (void)res;
  p_->set_terminator(static_cast<InstId>(p_->insts.size() - 1));
}
void IRBuilder::inst_ret_void() {
  ValueId res = p_->emit_inst_value(InstKind::Ret, type_void(), {});
  (void)res;
  p_->set_terminator(static_cast<InstId>(p_->insts.size() - 1));
}
ValueId IRBuilder::inst_phi(TypeId t, const std::vector<std::pair<ValueId, BlockId>>& incoming) {
  std::vector<ValueId> ops;
  ops.reserve(incoming.size() * 2);
  for (auto& pr : incoming) {
    ops.push_back(pr.first);
    ops.push_back(p_->make_block_value(p_->current_func, pr.second));
  }
  return p_->emit_inst_value(InstKind::Phi, t, ops);
}

ValueId IRBuilder::inst_call(ValueId callee, const std::vector<ValueId>& args, TypeId ret_type) {
  std::vector<ValueId> ops;
  ops.reserve(1 + args.size());
  ops.push_back(callee);
  ops.insert(ops.end(), args.begin(), args.end());
  return p_->emit_inst_value(InstKind::Call, ret_type, ops);
}

// -------- Introspection --------
const IRType& IRBuilder::type(TypeId t) const { return p_->type_cref(t); }
const IRValue& IRBuilder::value(ValueId v) const { return p_->value_cref(v); }

} // namespace see
