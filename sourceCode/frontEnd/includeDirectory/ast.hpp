#pragma once

#include <cstdint>
#include <cstddef>
#include <string_view>

#include "token.hpp"

namespace see {

// ---------- Non-owning slice (view over contiguous memory) ----------
template<typename T>
struct Slice {
  const T*  data{nullptr};
  std::uint32_t size{0};

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

// ---------- Node IDs (stable handles into arenas) ----------
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
  Q_Restrict  = 1u << 2, // C/C11
  Q_Atomic    = 1u << 3  // C11 _Atomic (as qualifier form)
};

enum : std::uint32_t {
  S_None         = 0,
  S_Extern       = 1u << 0,
  S_Static       = 1u << 1,
  S_Register     = 1u << 2,
  S_ThreadLocal  = 1u << 3, // _Thread_local / thread_local
  S_Inline       = 1u << 4
};

enum : std::uint32_t {
  F_None         = 0,
  F_Variadic     = 1u << 0,
  F_Noexcept     = 1u << 1, // C++ only
};

// ---------- Types ----------
enum class TypeKind : std::uint8_t {
  Invalid = 0,
  Primitive,      // see::Prim (int, double, ...)
  Pointer,        // ptr to another TypeId
  Array,          // arr of element TypeId, with optional length
  Function,       // return type + params
  Qualified,      // qualifiers applied on a base TypeId
  Named,          // typedef/using or tag name reference (not resolved yet)
  Record,         // struct/union definition
  Enum,           // enum definition
};

enum class RecordTag : std::uint8_t { Struct, Union };

struct Param {
  std::string_view name; // may be empty for unnamed parameter
  TypeId           type{kInvalidId};
  SourceRange      where{};
};

struct Field {
  std::string_view name;
  TypeId           type{kInvalidId};
  // Optional bit-field width; kInvalidId if not a bit-field (width as ExprId)
  ExprId           bit_width{kInvalidId};
  SourceRange      where{};
};

struct Enumerator {
  std::string_view name;
  // Optional value; kInvalidId if implicit
  ExprId           value{kInvalidId};
  SourceRange      where{};
};

struct Type {
  TypeKind    kind{TypeKind::Invalid};
  std::uint32_t flags{0}; // qualifiers (Q_*) for Qualified; F_* for Function

  SourceRange where{};

  union {
    // Primitive
    Prim primitive;

    // Pointer
    struct {
      TypeId  pointee;
    } ptr;

    // Array
    struct {
      TypeId  elem;
      // If fixed-length array, length_expr is an ExprId to an integral constant.
      // If kInvalidId, then it's unsized (as in 'int a[]') or VLA (depending on frontend rules).
      ExprId  length_expr;
    } arr;

    // Function
    struct {
      TypeId        ret;
      Slice<Param>  params;
    } func;

    // Qualified
    struct {
      TypeId  base;
    } qual;

    // Named (typedef, using, or tag reference by name)
    struct {
      std::string_view name;
    } named;

    // Record
    struct {
      RecordTag    tag;
      std::string_view name;     // may be empty for anonymous
      Slice<Field> fields;       // empty if forward-declared
    } rec;

    // Enum
    struct {
      std::string_view name;     // may be empty for anonymous
      Slice<Enumerator> enums;   // empty if forward-declared
    } enm;
  } as{};
};

// ---------- Expressions ----------
enum class ExprKind : std::uint8_t {
  Invalid = 0,
  // Primaries
  Identifier,        // name; (binding to DeclId is resolver’s job)
  IntegerLit,        // lexeme as view (parser may also compute value)
  FloatingLit,
  CharLit,
  StringLit,         // handles u8/u/U/L prefixed too (parser preserves prefix separately if needed)
  Paren,             // (expr) — preserves source structure

  // Postfix
  Call,              // callee(expr), args[]
  Index,             // base[idx]
  Member,            // base.member
  PtrMember,         // base->member
  PostInc,           // expr++
  PostDec,           // expr--

  // Unary
  PreInc,            // ++expr
  PreDec,            // --expr
  AddressOf,         // &expr
  Deref,             // *expr
  Plus,              // +expr
  Minus,             // -expr
  BitNot,            // ~expr
  LogNot,            // !expr
  SizeofExpr,        // sizeof(expr)
  SizeofType,        // sizeof(type)
  AlignofExpr,       // alignof(expr)
  AlignofType,       // alignof(type)
  CStyleCast,        // (type)expr
  CppCast,           // static_cast<T>(expr) etc. (record the cast token if needed)

  // Binary
  Mul, Div, Mod,
  Add, Sub,
  Shl, Shr,
  Lt, Le, Gt, Ge,
  Eq, Ne,
  BitAnd, BitXor, BitOr,
  LogAnd, LogOr,
  Assign,            // =
  AddAssign, SubAssign, MulAssign, DivAssign, ModAssign,
  ShlAssign, ShrAssign, AndAssign, XorAssign, OrAssign,

  // Ternary
  Conditional,       // cond ? then : else

  // Misc
  Comma              // a , b
};

struct Arg {
  ExprId expr{kInvalidId};
  SourceRange where{};
};

struct MemberRef {
  ExprId          base{kInvalidId};
  std::string_view name;
  SourceRange     where{};
};

struct Expr {
  ExprKind    kind{ExprKind::Invalid};
  SourceRange where{};

  union {
    // Identifier
    struct {
      std::string_view name;
      // Optional binding filled during resolution; not required at parse time.
      DeclId           binding{kInvalidId};
    } ident;

    // Literals
    struct { std::string_view text; } int_lit;
    struct { std::string_view text; } float_lit;
    struct { std::string_view text; } char_lit;
    struct { std::string_view text; } str_lit;

    // Paren
    struct { ExprId sub{kInvalidId}; } paren;

    // Call
    struct { ExprId callee{kInvalidId}; Slice<Arg> args; } call;

    // Index
    struct { ExprId base{kInvalidId}; ExprId index{kInvalidId}; } index;

    // Member / PtrMember
    struct { ExprId base{kInvalidId}; std::string_view name; } member;

    // Unary
    struct { ExprId sub{kInvalidId}; } unary;

    // Binary
    struct { ExprId lhs{kInvalidId}; ExprId rhs{kInvalidId}; } bin;

    // Assignment (reuse bin layout)
    struct { ExprId lhs2{kInvalidId}; ExprId rhs2{kInvalidId}; } asg;

    // Conditional
    struct { ExprId cond{kInvalidId}; ExprId then_e{kInvalidId}; ExprId else_e{kInvalidId}; } cond;

    // Casts & sizeof/alignof
    struct { ExprId subE{kInvalidId}; TypeId asT{kInvalidId}; } cast;      // C-style / C++ cast share this shape
    struct { ExprId subE2{kInvalidId}; } sizeof_align_expr;
    struct { TypeId asT2{kInvalidId}; } sizeof_align_type;
  } as{};
};

// ---------- Statements ----------
enum class StmtKind : std::uint8_t {
  Invalid = 0,
  Null,               // lone ';'
  ExprStmt,
  DeclStmt,
  Compound,           // { stmts... }
  If,
  While,
  DoWhile,
  For,                // classic for(init; cond; iter) body
  Switch,
  Case,
  Default,
  Break,
  Continue,
  Return,
  Goto,
  Label              // name: stmt
};

struct StmtList {
  const StmtId*     data{nullptr};
  std::uint32_t     size{0};
};

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
    struct { ExprId expr{kInvalidId}; } exprs;         // ExprStmt
    struct { DeclId decl{kInvalidId}; } decls;         // DeclStmt
    struct { StmtList list{}; } compound;              // Compound
    IfStmt      iff;
    WhileStmt   whil;
    DoWhileStmt dowhil;
    ForStmt     forr;
    SwitchStmt  swtch;
    CaseStmt    kase;
    DefaultStmt dflt;
    struct { ExprId expr{kInvalidId}; } ret;           // Return
    LabelStmt   label;
  } as{};
};

// ---------- Declarations ----------
enum class DeclKind : std::uint8_t {
  Invalid = 0,
  Var,
  Func,
  Typedef,
  Tag,        // struct/union/enum tag declaration/definition
};

struct Init {
  // For C-style initializers, you may extend this into aggregate forms later.
  ExprId        expr{kInvalidId};
  SourceRange   where{};
};

struct VarDecl {
  std::string_view name;
  TypeId           type{kInvalidId};
  std::uint32_t    storage{S_None}; // S_*
  bool             is_definition{false};
  Init             init{};          // kInvalidId expr if absent
};

struct FuncDecl {
  std::string_view name;
  // The function type (TypeKind::Function). Return type/params live there.
  TypeId           type{kInvalidId};
  std::uint32_t    storage{S_None}; // extern/static/inline/thread_local
  // If function definition, body != kInvalidId; otherwise declaration only.
  StmtId           body{kInvalidId};
};

struct TypedefDecl {
  std::string_view name;
  TypeId           aliased{kInvalidId};
};

struct TagDecl {
  // A struct/union/enum tag declaration or definition.
  // For definition, the corresponding Type node (Record/Enum) should be created
  // and linked through 'type'.
  std::string_view name; // may be empty for anonymous
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
  Slice<Decl>  decls;  // top-level declarations (order preserved)
  SourceRange  where{};
};

// ---------- Utility: Small helpers to check node kinds ----------
inline bool is_record(TypeKind k) noexcept { return k == TypeKind::Record; }
inline bool is_enum(TypeKind k)   noexcept { return k == TypeKind::Enum; }
inline bool is_func(TypeKind k)   noexcept { return k == TypeKind::Function; }
inline bool is_ptr(TypeKind k)    noexcept { return k == TypeKind::Pointer; }
inline bool is_array(TypeKind k)  noexcept { return k == TypeKind::Array; }
inline bool is_prim(TypeKind k)   noexcept { return k == TypeKind::Primitive; }

}
