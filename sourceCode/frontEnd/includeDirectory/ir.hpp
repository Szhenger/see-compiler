#pragma once
#include <cstdint>
#include <cstddef>
#include <string_view>

#include "token.hpp"  // for see::Arch

namespace see {

// --------------------------- Non-owning slice ---------------------------
template<typename T>
struct Slice {
  const T*      data{nullptr};
  std::uint32_t size{0};
  constexpr const T& operator[](std::uint32_t i) const noexcept { return data[i]; }
  constexpr bool empty() const noexcept { return size == 0; }
};

// --------------------------- Stable IDs ---------------------------
using TypeId   = std::uint32_t;  // index into type arena
using ValueId  = std::uint32_t;  // refers to constants, params, globals, inst results
using InstId   = std::uint32_t;  // index into instruction arena of its function
using BlockId  = std::uint32_t;  // index into function’s block arena
using FuncId   = std::uint32_t;  // index into module’s function arena
using GlobalId = std::uint32_t;  // index into module’s global arena
constexpr std::uint32_t kInvalidId = 0xFFFFFFFFu;

// --------------------------- Types ---------------------------
enum class IRTypeKind : std::uint8_t {
  Void = 0,
  Int,        // bitwidth (1,8,16,32,64,128)
  Float,      // 16,32,64,80 (x86 fp80), 128 (quad) — availability target-dependent
  Pointer,    // element type + addrspace (0 = generic)
  Function,   // return type + param types + vararg
};

struct IRType {
  IRTypeKind     kind{IRTypeKind::Void};
  std::uint32_t  bit_width{0};      // for Int/Float
  std::uint32_t  addr_space{0};     // for Pointer
  TypeId         elem{kInvalidId};  // for Pointer
  TypeId         ret{kInvalidId};   // for Function
  Slice<TypeId>  params{};          // for Function
  bool           is_vararg{false};  // for Function
};

// --------------------------- Values ---------------------------
enum class IRValueKind : std::uint8_t {
  Undef = 0,
  ConstInt,
  ConstFloat,
  Global,     // global object/function symbol
  Param,      // function parameter
  Inst,       // instruction result
};

struct IRConstInt {
  TypeId        type{kInvalidId}; // must be Int or Pointer-sized Int
  std::uint64_t lo{0};            // 128-bit integers can be supported later
  std::uint64_t hi{0};
  bool          is_signed{false};
};

struct IRConstFloat {
  TypeId        type{kInvalidId}; // Float type
  // Raw storage; builder is responsible for packing (e.g., via memcpy)
  std::uint64_t bits_lo{0};
  std::uint64_t bits_hi{0};
};

struct IRGlobal {
  std::string_view name;
  TypeId           type{kInvalidId}; // pointer-to or object type depending on model
  bool             is_function{false};
  bool             is_extern{true};
  bool             is_mutable{true};
};

struct IRParam {
  FuncId  func{kInvalidId};
  TypeId  type{kInvalidId};
  std::uint32_t index{0};
};

struct IRInstRef {
  FuncId  func{kInvalidId};
  BlockId block{kInvalidId};
  InstId  inst{kInvalidId};
};

struct IRValue {
  IRValueKind kind{IRValueKind::Undef};
  TypeId      type{kInvalidId}; // the value’s IR type

  union {
    IRConstInt   c_int;
    IRConstFloat c_flt;
    IRGlobal     global;
    IRParam      param;
    IRInstRef    inst;
  } as{};
};

// --------------------------- Instructions ---------------------------
// Many opcodes mirror LLVM-like semantics in a compact set.

enum class ICmpCond : std::uint8_t { EQ, NE, ULT, ULE, UGT, UGE, SLT, SLE, SGT, SGE };
enum class FCmpCond : std::uint8_t { OEQ, ONE, OLT, OLE, OGT, OGE, UEQ, UNE, ULT, ULE, UGT, UGE, ORD, UNO };

enum class CastKind : std::uint8_t {
  Bitcast,     // same bitwidth, different type
  ZExt,        // int -> wider int (zero-extend)
  SExt,        // int -> wider int (sign-extend)
  Trunc,       // int -> narrower int
  UIToFP,      // unsigned int -> float
  SIToFP,      // signed int -> float
  FPToUI,      // float -> unsigned int
  FPToSI,      // float -> signed int
  FPTrunc,     // float narrow
  FPExt,       // float widen
  PtrToInt,    // ptr -> int (same size)
  IntToPtr     // int -> ptr (same size)
};

enum class InstKind : std::uint16_t {
  // Memory & alloc
  Alloca,          // result: pointer to allocated stack object; op0 = type (in a Type pseudo-value)
  Load,            // op0 = ptr
  Store,           // op0 = value, op1 = ptr (no result)
  Gep,             // result: ptr; op0 = base ptr; op1.. = indices (int)

  // Integer arithmetic (wrap/overflow semantics defined by front-end)
  IAdd, ISub, IMul,
  UDiv, SDiv, URem, SRem,
  Shl, LShr, AShr,
  And, Or, Xor,

  // Floating arithmetic
  FAdd, FSub, FMul, FDiv, FRem,

  // Comparisons
  ICmp,            // op0 = lhs, op1 = rhs; imm = ICmpCond
  FCmp,            // op0 = lhs, op1 = rhs; imm = FCmpCond

  // Casts / conversions
  Cast,            // op0 = value; imm = CastKind

  // Control flow
  Br,              // op0 = target block
  CondBr,          // op0 = cond; op1 = then block; op2 = else block
  Ret,             // op0 = value (optional depending on function type)
  Phi,             // result; pairs of (value, incoming block) in operands

  // Calls
  Call,            // op0 = callee (ValueId, usually Global); op1.. = args
};

// Operand is simply a ValueId; blocks in control flow positions are encoded as “block-value”
// via special typed values the builder creates (or by side channels in the instruction).
struct IROperand {
  ValueId v{kInvalidId};
};

// Each instruction defines at most one result (void-typed instructions don’t produce a ValueId).
struct IRInst {
  InstKind      kind{InstKind::Br};
  TypeId        type{kInvalidId};     // result type if any (Void otherwise)
  Slice<IROperand> ops{};             // operands (ValueIds)
  std::uint32_t imm0{0};              // small immediate (e.g., cmp/cast condition)
  // For Phi, ops are (v0, block0, v1, block1, ...)
};

// --------------------------- CFG: Blocks / Functions / Module ---------------------------
struct IRBlock {
  std::string_view name;
  Slice<InstId>    insts;        // instruction ids in emission order
  InstId           terminator{kInvalidId}; // must be last inst and a terminator kind
};

struct IRFunction {
  std::string_view name;
  TypeId           type{kInvalidId};     // Function type
  Slice<BlockId>   blocks;               // basic blocks in RPO or insertion order
  Slice<ValueId>   params;               // parameter ValueIds
  bool             is_decl{true};        // true until a body is emitted
};

struct IRGlobalObj {
  std::string_view name;
  TypeId           type{kInvalidId};
  bool             is_function{false};
  bool             is_extern{true};
  bool             is_mutable{true};
  // Optional initializer would be modeled as a small IR or constant buffer later
};

struct IRModule {
  Arch             target{Arch::X86_64};
  Slice<IRType>    types;
  Slice<IRValue>   values;
  Slice<IRGlobalObj> globals;
  Slice<IRFunction> functions;
};

// =============================== IR BUILDER (PUBLIC API) ===============================
//
// Implemented in ir.cpp. Owns arenas; returns stable IDs and slices.

class IRBuilder {
public:
  explicit IRBuilder(Arch target);

  // -------- Types --------
  TypeId type_void();
  TypeId type_int(std::uint32_t bit_width);               // 1/8/16/32/64/128
  TypeId type_float(std::uint32_t bit_width);             // 16/32/64/80/128 (as supported)
  TypeId type_ptr(TypeId elem, std::uint32_t addrspace=0);
  TypeId type_func(TypeId ret, const std::vector<TypeId>& params, bool vararg=false);

  // -------- Globals --------
  GlobalId global_var(std::string_view name, TypeId type, bool is_extern, bool is_mutable);
  GlobalId global_func(std::string_view name, TypeId func_type, bool is_extern);

  // -------- Functions & blocks --------
  FuncId  func_begin(std::string_view name, TypeId func_type); // switches builder to this func
  void    func_set_decl(FuncId f, bool is_decl);
  ValueId param_value(FuncId f, std::uint32_t index);

  BlockId block_create(FuncId f, std::string_view name);
  void    block_set_insert_point(FuncId f, BlockId b);         // subsequent inst_* append here

  // -------- Constants --------
  ValueId const_int(TypeId t, std::uint64_t lo, std::uint64_t hi=0, bool is_signed=false);
  ValueId const_float_bits(TypeId t, std::uint64_t lo, std::uint64_t hi=0);

  // -------- Instructions (results return ValueId or kInvalidId if void) --------
  ValueId inst_alloca(TypeId obj_type);
  ValueId inst_load(ValueId ptr);
  void    inst_store(ValueId value, ValueId ptr);
  ValueId inst_gep(ValueId base_ptr, const std::vector<ValueId>& indices);

  // Integer arith
  ValueId inst_bin(InstKind k, ValueId a, ValueId b);          // IAdd/ISub/IMul/UDiv/SDiv/URem/SRem/Shl/LShr/AShr/And/Or/Xor
  // Floating arith
  ValueId inst_fbin(InstKind k, ValueId a, ValueId b);         // FAdd/FSub/FMul/FDiv/FRem

  // Comparisons
  ValueId inst_icmp(ICmpCond c, ValueId a, ValueId b);         // result is i1
  ValueId inst_fcmp(FCmpCond c, ValueId a, ValueId b);         // result is i1

  // Casts
  ValueId inst_cast(CastKind c, ValueId v, TypeId to_type);

  // Control flow
  void    inst_br(BlockId target);
  void    inst_condbr(ValueId cond, BlockId then_b, BlockId else_b);
  void    inst_ret(ValueId v);                                  // for non-void
  void    inst_ret_void();                                      // for void
  ValueId inst_phi(TypeId t, const std::vector<std::pair<ValueId, BlockId>>& incoming);

  // Calls
  ValueId inst_call(ValueId callee, const std::vector<ValueId>& args, TypeId ret_type);

  // -------- Finish --------
  IRModule finish();

  // -------- Introspection (read-only) --------
  const IRType&   type(TypeId t)   const;
  const IRValue&  value(ValueId v) const;

private:
  // Opaque state; defined in ir.cpp
  struct Impl;
  Impl* p_;
};

// ------------------------ Small helpers / predicates ------------------------
inline bool is_terminator(InstKind k) noexcept {
  switch (k) {
    case InstKind::Br: case InstKind::CondBr: case InstKind::Ret: return true;
    default: return false;
  }
}

inline bool is_int_type(const IRType& t)    noexcept { return t.kind == IRTypeKind::Int; }
inline bool is_float_type(const IRType& t)  noexcept { return t.kind == IRTypeKind::Float; }
inline bool is_ptr_type(const IRType& t)    noexcept { return t.kind == IRTypeKind::Pointer; }
inline bool is_func_type(const IRType& t)   noexcept { return t.kind == IRTypeKind::Function; }
inline bool is_void_type(const IRType& t)   noexcept { return t.kind == IRTypeKind::Void; }

} // namespace see
