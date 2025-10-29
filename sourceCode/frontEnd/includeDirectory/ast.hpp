#pragma once
#include <cstdint>
#include <cstddef>
#include <string_view>

#include "token.hpp" // for see::Prim

namespace see {

// ---------- Non-owning slice ----------
template<typename T>
struct Slice {
  const T*        data{nullptr};
  std::uint32_t   size{0};

  constexpr const T& operator[](std::uint32_t i) const noexcept { return data[i]; }
  constexpr bool empty() const noexcept { return size == 0; }
};

// ---------- Source locations ----------
struct SourcePos {
  std::uint32_t line{0};   // 1-based
  std::uint32_t column{0}; // 1-based
};

struct SourceRange {
  const char*   begin{nullptr}; // pointer into the original buffer
  const char*   end{nullptr};   // one-past-end
  SourcePos     start{};        // optional human-readable position
};

// ---------- Node IDs ----------
using TypeId = std::uint32_t;
using ExprId = std::uint32_t;
using StmtId = std::uint32_t;
using DeclId = std::uint32_t;
constexpr std::uint32_t kInvalidId = 0xFFFFFFFFu;

// ---------- Qualifiers / storage / function flags ----------
enum : std::uint32_t {
  Q_None      = 0,
  Q_Const     = 1u << 0,
  Q_Volatile  = 1u << 1,
  Q_Restrict  = 1u << 2,
  Q_Atomic    = 1u << 3
};

enum : std::uint32_t {
  S_None         = 0,
  S_Extern       = 1u << 0,
  S_Static       = 1u << 1,
  S_Register     = 1u << 2,
  S_ThreadLocal  = 1u << 3,
  S_Inline       = 1u << 4
};

enum : std::uint32_t {
  F_None         = 0,
  F_Variadic     = 1u << 0,
  F_Noexcept     = 1u << 1,
};

// ---------- Types ----------
enum class TypeKind : std::uint8_t {
  Invalid = 0,
  Primitive,
  Pointer,
  Array,
  Function,
  Qualified,
  Named,
  Record,
  Enum,
};

enum class RecordTag : std::uint8_t { Struct, Union };

struct Param {
  std::string_view name;
  TypeId           type{kInvalidId};
  SourceRange      where{};
};

struct Field {
  std::string_view name;
  TypeId           type{kInvalidId};
  ExprId           bit_width{kInvalidId}; // if bit-field; else kInvalidId
  SourceRange      where{};
};

struct Enumerator {
  std::string_view name;
  ExprId           value{kInvalidId}; // optional explicit value
  SourceRange      where{};
};

struct Type {
  TypeKind      kind{TypeKind::Invalid};
  std::uint32_t flags{0}; // Q_* for Qualified; F_* for Function
  SourceRange   where{};

  union {
    // Primitive
    Prim primitive;

    // Pointer
    struct { TypeId pointee; } ptr;

    // Array
    struct { TypeId elem; ExprId length_expr; } arr;

    // Function
    struct { TypeId ret; Slice<Param> params; } func;

    // Qualified
    struct { TypeId base; } qual;

    // Named
    struct { std::string_view name; } named;

    // Record
    struct { RecordTag tag; std::string_view name; Slice<Field> fields; } rec;

    // Enum
    struct { std::string_view name; Slice<Enumerator> enums; } enm;
  } as{};
};

// ---------- Expressions ----------
enum class ExprKind : std::uint8_t {
  Invalid = 0,
  // Primaries
  Identifier,
  IntegerLit,
  FloatingLit,
  CharLit,
  StringLit,
  Paren,

  // Postfix
  Call,
  Index,
  Member,
  PtrMember,
  PostInc,
  PostDec,

  // Unary
  PreInc,
  PreDec,
  AddressOf,
  Deref,
  Plus,
  Minus,
  BitNot,
  LogNot,
  SizeofExpr,
  SizeofType,
  AlignofExpr,
  AlignofType,
  CStyleCast,
  CppCast,

  // Binary
  Mul, Div, Mod,
  Add, Sub,
  Shl, Shr,
  Lt, Le, Gt, Ge,
  Eq, Ne,
  BitAnd, BitXor, BitOr,
  LogAnd, LogOr,
  Assign,
  AddAssign, SubAssign, MulAssign, DivAssign, ModAssign,
  ShlAssign, ShrAssign, AndAssign, XorAssign, OrAssign,

  // Ternary
  Conditional,

  // Misc
  Comma
};

struct Arg {
  ExprId       expr{kInvalidId};
  SourceRange  where{};
};

struct Expr {
  ExprKind    kind{ExprKind::Invalid};
  SourceRange where{};

  union {
    struct { std::string_view name; DeclId binding; } ident;

    // Literals
    struct { std::string_view text; } int_lit;
    struct { std::string_view text; } float_lit;
    struct { std::string_view text; } char_lit;
    struct { std::string_view text; } str_lit;

    // Paren
    struct { ExprId sub; } paren;

    // Call
    struct { ExprId callee; Slice<Arg> args; } call;

    // Index
    struct { ExprId base; ExprId index; } index;

    // Member / PtrMember (share layout)
    struct { ExprId base; std::string_view name; } member;

    // Unary
    struct { ExprId sub; } unary;

    // Binary
    struct { ExprId lhs; ExprId rhs; } bin;

    // Assignment
    struct { ExprId lhs2; ExprId rhs2; } asg;

    // Conditional
    struct { ExprId cond; ExprId then_e; ExprId else_e; } cond;

    // Casts & sizeof/alignof
    struct { ExprId subE; TypeId asT; } cast;
    struct { ExprId subE2; } sizeof_align_expr;
    struct { TypeId asT2; } sizeof_align_type;
  } as{};
};

// ---------- Statements ----------
enum class StmtKind : std::uint8_t {
  Invalid = 0,
  Null,
  ExprStmt,
  DeclStmt,
  Compound,
  If,
  While,
  DoWhile,
  For,
  Switch,
  Case,
  Default,
  Break,
  Continue,
  Return,
  Goto,
  Label
};

struct StmtList { const StmtId* data{nullptr}; std::uint32_t size{0}; };

struct IfStmt { ExprId cond{kInvalidId}; StmtId then_s{kInvalidId}; StmtId else_s{kInvalidId}; };
struct WhileStmt { ExprId cond{kInvalidId}; StmtId body{kInvalidId}; };
struct DoWhileStmt { StmtId body{kInvalidId}; ExprId cond{kInvalidId}; };
struct ForStmt { StmtId init{kInvalidId}; ExprId cond{kInvalidId}; ExprId iter{kInvalidId}; StmtId body{kInvalidId}; };
struct SwitchStmt { ExprId expr{kInvalidId}; StmtId body{kInvalidId}; };
struct CaseStmt { ExprId value{kInvalidId}; StmtId body{kInvalidId}; };
struct DefaultStmt { StmtId body{kInvalidId}; };
struct LabelStmt { std::string_view name; StmtId body{kInvalidId}; };

struct Stmt {
  StmtKind    kind{StmtKind::Invalid};
  SourceRange where{};

  union {
    struct { ExprId expr; } exprs;     // ExprStmt
    struct { DeclId decl; } decls;     // DeclStmt
    struct { StmtList list; } compound;
    IfStmt      iff;
    WhileStmt   whil;
    DoWhileStmt dowhil;
    ForStmt     forr;
    SwitchStmt  swtch;
    CaseStmt    kase;
    DefaultStmt dflt;
    struct { ExprId expr; } ret;
    LabelStmt   label;
  } as{};
};

// ---------- Declarations ----------
enum class DeclKind : std::uint8_t { Invalid = 0, Var, Func, Typedef, Tag };

struct Init { ExprId expr{kInvalidId}; SourceRange where{}; };

struct VarDecl {
  std::string_view name;
  TypeId           type{kInvalidId};
  std::uint32_t    storage{S_None};
  bool             is_definition{false};
  Init             init{};
};

struct FuncDecl {
  std::string_view name;
  TypeId           type{kInvalidId}; // TypeKind::Function
  std::uint32_t    storage{S_None};
  StmtId           body{kInvalidId}; // if definition
};

struct TypedefDecl { std::string_view name; TypeId aliased{kInvalidId}; };

struct TagDecl {
  std::string_view name;
  TypeId           type{kInvalidId}; // points to Record/Enum Type
};

struct Decl {
  DeclKind    kind{DeclKind::Invalid};
  SourceRange where{};

  union {
    VarDecl     var;
    FuncDecl    func;
    TypedefDecl tdef;
    TagDecl     tag;
  } as{};
};

// ---------- Translation unit ----------
struct TranslationUnit {
  Slice<Decl> decls;
  SourceRange where{};
};

// ---------- Kind predicates ----------
inline bool is_record(TypeKind k) noexcept { return k == TypeKind::Record; }
inline bool is_enum(TypeKind k)   noexcept { return k == TypeKind::Enum; }
inline bool is_func(TypeKind k)   noexcept { return k == TypeKind::Function; }
inline bool is_ptr(TypeKind k)    noexcept { return k == TypeKind::Pointer; }
inline bool is_array(TypeKind k)  noexcept { return k == TypeKind::Array; }
inline bool is_prim(TypeKind k)   noexcept { return k == TypeKind::Primitive; }

// ============================================================================
//                           AST BUILDER (PUBLIC API)
// ============================================================================
//
// All methods are implemented in ast.cpp. This subset matches what parser.cpp calls.

class ASTBuilder {
public:
  // String interning (stable views)
  std::string_view intern(std::string_view s);
  std::string_view intern(const char* s);

  // Types
  TypeId type_primitive(Prim prim, SourceRange w = {});
  TypeId type_pointer(TypeId to, SourceRange w = {});
  TypeId type_array(TypeId elem, ExprId length_expr = kInvalidId, SourceRange w = {});
  TypeId type_function(TypeId ret, const std::vector<Param>& params,
                       std::uint32_t flags = F_None, SourceRange w = {});
  TypeId type_qualified(TypeId base, std::uint32_t qual_flags, SourceRange w = {});
  TypeId type_named(std::string_view name, SourceRange w = {});
  TypeId type_record(RecordTag tag, std::string_view name,
                     const std::vector<Field>& fields, SourceRange w = {});
  TypeId type_enum(std::string_view name, const std::vector<Enumerator>& enums,
                   SourceRange w = {});

  // Expressions
  ExprId expr_identifier(std::string_view name, SourceRange w = {});
  ExprId expr_integer(std::string_view text, SourceRange w = {});
  ExprId expr_floating(std::string_view text, SourceRange w = {});
  ExprId expr_string(std::string_view text, SourceRange w = {});
  ExprId expr_char(std::string_view text, SourceRange w = {});
  ExprId expr_paren(ExprId sub, SourceRange w = {});
  ExprId expr_unary(ExprKind k, ExprId sub, SourceRange w = {});
  ExprId expr_binary(ExprKind k, ExprId lhs, ExprId rhs, SourceRange w = {});
  ExprId expr_assign(ExprKind k, ExprId lhs, ExprId rhs, SourceRange w = {});
  ExprId expr_call(ExprId callee, const std::vector<Arg>& args, SourceRange w = {});
  ExprId expr_index(ExprId base, ExprId idx, SourceRange w = {});
  ExprId expr_member(bool ptr, ExprId base, std::string_view name, SourceRange w = {});
  ExprId expr_conditional(ExprId c, ExprId t, ExprId f, SourceRange w = {});
  ExprId expr_c_style_cast(TypeId to, ExprId sub, SourceRange w = {});
  ExprId expr_sizeof_type(TypeId t, SourceRange w = {});
  ExprId expr_sizeof_expr(ExprId sub, SourceRange w = {});

  // Statements
  StmtId stmt_null(SourceRange w = {});
  StmtId stmt_expr(ExprId e, SourceRange w = {});
  StmtId stmt_decl(DeclId d, SourceRange w = {});
  StmtId stmt_compound(const std::vector<StmtId>& stmts, SourceRange w = {});
  StmtId stmt_if(ExprId cond, StmtId thn, StmtId els, SourceRange w = {});
  StmtId stmt_while(ExprId cond, StmtId body, SourceRange w = {});
  StmtId stmt_dowhile(StmtId body, ExprId cond, SourceRange w = {});
  StmtId stmt_for(StmtId init, ExprId cond, ExprId iter, StmtId body, SourceRange w = {});
  StmtId stmt_switch(ExprId e, StmtId body, SourceRange w = {});
  StmtId stmt_case(ExprId val, StmtId body, SourceRange w = {});
  StmtId stmt_default(StmtId body, SourceRange w = {});
  StmtId stmt_return(ExprId e, SourceRange w = {});
  StmtId stmt_label(std::string_view name, StmtId body, SourceRange w = {});

  // Declarations (+ TU)
  DeclId decl_var(std::string_view name, TypeId type,
                  std::uint32_t storage, bool is_definition,
                  ExprId init_expr, SourceRange w = {});
  DeclId decl_func(std::string_view name, TypeId func_type,
                   std::uint32_t storage, StmtId body, SourceRange w = {});
  DeclId decl_typedef(std::string_view name, TypeId aliased, SourceRange w = {});
  DeclId decl_tag(std::string_view name, TypeId type, SourceRange w = {});
  void   push_toplevel(DeclId d);
  TranslationUnit finish();

  // Read-only access by ID (parser uses `type()` to inspect kind).
  const Type& type(TypeId id) const;
  const Expr& expr(ExprId id) const;
  const Stmt& stmt(StmtId id) const;
  const Decl& decl(DeclId id) const;

  // (Optional) expose a generic push for statements if needed by ad-hoc constructs
  // StmtId push_stmt(Stmt&& s);   // uncomment if your parser calls it directly
};

} // namespace see
