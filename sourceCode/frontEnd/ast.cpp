#include "ast.hpp"

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <cassert>
#include <cstring> // memcpy

namespace see {

// ----------------------------- helpers -----------------------------

static inline SourceRange where(const char* b, const char* e,
                                std::uint32_t line=0, std::uint32_t col=0) {
  SourceRange r;
  r.begin = b; r.end = e;
  r.start.line = line; r.start.column = col;
  return r;
}

// A segmented arena that returns contiguous slices with stable addresses.
// Each appended "run" is placed in its own heap block (unique_ptr<T[]>),
// ensuring that Slice<T>.data never moves.
template<typename T>
class SegArena {
public:
  Slice<T> copy_into(const T* src, std::uint32_t n) {
    if (n == 0) return Slice<T>{nullptr, 0};
    std::unique_ptr<T[]> block(new T[n]);
    for (std::uint32_t i = 0; i < n; ++i) block[i] = src[i];
    T* ptr = block.get();
    blocks_.emplace_back(std::move(block));
    return Slice<T>{ptr, n};
  }

  Slice<T> copy_into(std::initializer_list<T> init) {
    return copy_into(init.begin(), static_cast<std::uint32_t>(init.size()));
  }

  // Construct in-place from a std::vector<T>.
  Slice<T> copy_into(const std::vector<T>& v) {
    return copy_into(v.data(), static_cast<std::uint32_t>(v.size()));
  }

private:
  std::vector<std::unique_ptr<T[]>> blocks_;
};

// A minimal string interner that returns stable std::string_view references.
// Interned storage lives for the builder's lifetime.
class StringInterner {
public:
  std::string_view intern(std::string_view s) {
    auto it = set_.find(s);
    if (it != set_.end()) return *it;
    auto inserted = set_.emplace(std::string(s)); // copy into set storage
    return *(inserted.first);
  }

  std::string_view intern(const char* s) {
    return intern(std::string_view{s});
  }

private:
  struct ViewHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept {
      return std::hash<std::string_view>{}(s);
    }
    size_t operator()(const std::string& s) const noexcept {
      return std::hash<std::string_view>{}(std::string_view{s});
    }
  };
  struct ViewEq {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const noexcept { return a == b; }
    bool operator()(const std::string& a, const std::string& b) const noexcept { return a == b; }
    bool operator()(const std::string& a, std::string_view b) const noexcept { return std::string_view{a} == b; }
    bool operator()(std::string_view a, const std::string& b) const noexcept { return a == std::string_view{b}; }
  };

  // We store std::string in the set so the characters are owned and stable.
  std::unordered_set<std::string, ViewHash, ViewEq> set_;
};

// ----------------------------- ASTBuilder -----------------------------

class ASTBuilder {
public:
  // ---- string interning ----
  std::string_view intern(std::string_view s) { return strings_.intern(s); }
  std::string_view intern(const char* s)      { return strings_.intern(s); }

  // ---- Type constructors ----
  TypeId type_primitive(Prim prim, SourceRange w = {}) {
    Type t;
    t.kind = TypeKind::Primitive;
    t.where = w;
    t.as.primitive = prim;
    return push_type(std::move(t));
  }

  TypeId type_pointer(TypeId to, SourceRange w = {}) {
    Type t;
    t.kind = TypeKind::Pointer;
    t.where = w;
    t.as.ptr.pointee = to;
    return push_type(std::move(t));
  }

  TypeId type_array(TypeId elem, ExprId length_expr = kInvalidId, SourceRange w = {}) {
    Type t;
    t.kind = TypeKind::Array;
    t.where = w;
    t.as.arr.elem = elem;
    t.as.arr.length_expr = length_expr;
    return push_type(std::move(t));
  }

  TypeId type_function(TypeId ret, const std::vector<Param>& params,
                       std::uint32_t flags = F_None, SourceRange w = {}) {
    Type t;
    t.kind = TypeKind::Function;
    t.where = w;
    t.flags = flags;
    t.as.func.ret = ret;
    t.as.func.params = params_arena_.copy_into(params);
    return push_type(std::move(t));
  }

  TypeId type_qualified(TypeId base, std::uint32_t qual_flags, SourceRange w = {}) {
    Type t;
    t.kind = TypeKind::Qualified;
    t.where = w;
    t.flags = qual_flags; // Q_*
    t.as.qual.base = base;
    return push_type(std::move(t));
  }

  TypeId type_named(std::string_view name, SourceRange w = {}) {
    Type t;
    t.kind = TypeKind::Named;
    t.where = w;
    t.as.named.name = intern(name);
    return push_type(std::move(t));
  }

  TypeId type_record(RecordTag tag, std::string_view name,
                     const std::vector<Field>& fields, SourceRange w = {}) {
    Type t;
    t.kind = TypeKind::Record;
    t.where = w;
    t.as.rec.tag = tag;
    t.as.rec.name = intern(name);
    t.as.rec.fields = fields_arena_.copy_into(fields);
    return push_type(std::move(t));
  }

  TypeId type_enum(std::string_view name, const std::vector<Enumerator>& enums,
                   SourceRange w = {}) {
    Type t;
    t.kind = TypeKind::Enum;
    t.where = w;
    t.as.enm.name = intern(name);
    t.as.enm.enums = enums_arena_.copy_into(enums);
    return push_type(std::move(t));
  }

  // ---- Expr constructors (selected; extend as needed) ----
  ExprId expr_identifier(std::string_view name, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::Identifier;
    e.where = w;
    e.as.ident.name = intern(name);
    e.as.ident.binding = kInvalidId;
    return push_expr(std::move(e));
  }

  ExprId expr_integer(std::string_view text, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::IntegerLit;
    e.where = w;
    e.as.int_lit.text = intern(text);
    return push_expr(std::move(e));
  }

  ExprId expr_floating(std::string_view text, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::FloatingLit;
    e.where = w;
    e.as.float_lit.text = intern(text);
    return push_expr(std::move(e));
  }

  ExprId expr_string(std::string_view text, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::StringLit;
    e.where = w;
    e.as.str_lit.text = intern(text);
    return push_expr(std::move(e));
  }

  ExprId expr_char(std::string_view text, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::CharLit;
    e.where = w;
    e.as.char_lit.text = intern(text);
    return push_expr(std::move(e));
  }

  ExprId expr_paren(ExprId sub, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::Paren;
    e.where = w;
    e.as.paren.sub = sub;
    return push_expr(std::move(e));
  }

  ExprId expr_unary(ExprKind k, ExprId sub, SourceRange w = {}) {
    assert(k == ExprKind::PreInc || k == ExprKind::PreDec ||
           k == ExprKind::AddressOf || k == ExprKind::Deref ||
           k == ExprKind::Plus || k == ExprKind::Minus ||
           k == ExprKind::BitNot || k == ExprKind::LogNot ||
           k == ExprKind::SizeofExpr || k == ExprKind::AlignofExpr);
    Expr e;
    e.kind = k;
    e.where = w;
    e.as.unary.sub = sub;
    return push_expr(std::move(e));
  }

  ExprId expr_binary(ExprKind k, ExprId lhs, ExprId rhs, SourceRange w = {}) {
    // Accept any binary/assign op kind listed in ExprKind
    Expr e;
    e.kind = k;
    e.where = w;
    e.as.bin.lhs = lhs;
    e.as.bin.rhs = rhs;
    return push_expr(std::move(e));
  }

  ExprId expr_assign(ExprKind k, ExprId lhs, ExprId rhs, SourceRange w = {}) {
    // k one of Assign/AddAssign/SubAssign/... etc.
    Expr e;
    e.kind = k;
    e.where = w;
    e.as.asg.lhs2 = lhs;
    e.as.asg.rhs2 = rhs;
    return push_expr(std::move(e));
  }

  ExprId expr_call(ExprId callee, const std::vector<Arg>& args, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::Call;
    e.where = w;
    e.as.call.callee = callee;
    e.as.call.args = args_arena_.copy_into(args);
    return push_expr(std::move(e));
  }

  ExprId expr_index(ExprId base, ExprId idx, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::Index;
    e.where = w;
    e.as.index.base = base;
    e.as.index.index = idx;
    return push_expr(std::move(e));
  }

  ExprId expr_member(bool ptr, ExprId base, std::string_view name, SourceRange w = {}) {
    Expr e;
    e.kind = ptr ? ExprKind::PtrMember : ExprKind::Member;
    e.where = w;
    e.as.member.base = base;
    e.as.member.name = intern(name);
    return push_expr(std::move(e));
  }

  ExprId expr_conditional(ExprId c, ExprId t, ExprId f, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::Conditional;
    e.where = w;
    e.as.cond.cond = c;
    e.as.cond.then_e = t;
    e.as.cond.else_e = f;
    return push_expr(std::move(e));
  }

  ExprId expr_c_style_cast(TypeId to, ExprId sub, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::CStyleCast;
    e.where = w;
    e.as.cast.asT = to;
    e.as.cast.subE = sub;
    return push_expr(std::move(e));
  }

  ExprId expr_sizeof_type(TypeId t, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::SizeofType;
    e.where = w;
    e.as.sizeof_align_type.asT2 = t;
    return push_expr(std::move(e));
  }

  ExprId expr_sizeof_expr(ExprId sub, SourceRange w = {}) {
    Expr e;
    e.kind = ExprKind::SizeofExpr;
    e.where = w;
    e.as.sizeof_align_expr.subE2 = sub;
    return push_expr(std::move(e));
  }

  // ---- Stmt constructors ----
  StmtId stmt_null(SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::Null; s.where = w;
    return push_stmt(std::move(s));
  }

  StmtId stmt_expr(ExprId e, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::ExprStmt; s.where = w;
    s.as.exprs.expr = e;
    return push_stmt(std::move(s));
  }

  StmtId stmt_decl(DeclId d, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::DeclStmt; s.where = w;
    s.as.decls.decl = d;
    return push_stmt(std::move(s));
  }

  StmtId stmt_compound(const std::vector<StmtId>& stmts, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::Compound; s.where = w;
    s.as.compound.list = stmt_list_arena_.copy_into(stmts);
    return push_stmt(std::move(s));
  }

  StmtId stmt_if(ExprId cond, StmtId thn, StmtId els, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::If; s.where = w;
    s.as.iff.cond = cond; s.as.iff.then_s = thn; s.as.iff.else_s = els;
    return push_stmt(std::move(s));
  }

  StmtId stmt_while(ExprId cond, StmtId body, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::While; s.where = w;
    s.as.whil.cond = cond; s.as.whil.body = body;
    return push_stmt(std::move(s));
  }

  StmtId stmt_dowhile(StmtId body, ExprId cond, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::DoWhile; s.where = w;
    s.as.dowhil.body = body; s.as.dowhil.cond = cond;
    return push_stmt(std::move(s));
  }

  StmtId stmt_for(StmtId init, ExprId cond, ExprId iter, StmtId body, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::For; s.where = w;
    s.as.forr.init = init; s.as.forr.cond = cond; s.as.forr.iter = iter; s.as.forr.body = body;
    return push_stmt(std::move(s));
  }

  StmtId stmt_switch(ExprId e, StmtId body, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::Switch; s.where = w;
    s.as.swtch.expr = e; s.as.swtch.body = body;
    return push_stmt(std::move(s));
  }

  StmtId stmt_case(ExprId val, StmtId body, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::Case; s.where = w;
    s.as.kase.value = val; s.as.kase.body = body;
    return push_stmt(std::move(s));
  }

  StmtId stmt_default(StmtId body, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::Default; s.where = w;
    s.as.dflt.body = body;
    return push_stmt(std::move(s));
  }

  StmtId stmt_return(ExprId e, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::Return; s.where = w;
    s.as.ret.expr = e;
    return push_stmt(std::move(s));
  }

  StmtId stmt_label(std::string_view name, StmtId body, SourceRange w = {}) {
    Stmt s; s.kind = StmtKind::Label; s.where = w;
    s.as.label.name = intern(name); s.as.label.body = body;
    return push_stmt(std::move(s));
  }

  // ---- Decl constructors ----
  DeclId decl_var(std::string_view name, TypeId type,
                  std::uint32_t storage, bool is_definition,
                  ExprId init_expr, SourceRange w = {}) {
    Decl d; d.kind = DeclKind::Var; d.where = w;
    d.as.var.name = intern(name);
    d.as.var.type = type;
    d.as.var.storage = storage;
    d.as.var.is_definition = is_definition;
    d.as.var.init.expr = init_expr;
    return push_decl(std::move(d));
  }

  DeclId decl_func(std::string_view name, TypeId func_type,
                   std::uint32_t storage, StmtId body, SourceRange w = {}) {
    Decl d; d.kind = DeclKind::Func; d.where = w;
    d.as.func.name = intern(name);
    d.as.func.type = func_type;
    d.as.func.storage = storage;
    d.as.func.body = body;
    return push_decl(std::move(d));
  }

  DeclId decl_typedef(std::string_view name, TypeId aliased, SourceRange w = {}) {
    Decl d; d.kind = DeclKind::Typedef; d.where = w;
    d.as.tdef.name = intern(name);
    d.as.tdef.aliased = aliased;
    return push_decl(std::move(d));
  }

  DeclId decl_tag(std::string_view name, TypeId type, SourceRange w = {}) {
    Decl d; d.kind = DeclKind::Tag; d.where = w;
    d.as.tag.name = intern(name);
    d.as.tag.type = type;
    return push_decl(std::move(d));
  }

  void push_toplevel(DeclId d) {
    toplevel_.push_back(d);
  }

  // Build a TranslationUnit view. Returned slices reference memory owned by *this*.
  TranslationUnit finish() {
    // Materialize top-level declarations into a contiguous array.
    // We need Decl objects, not just IDs. Consumers typically want Decl slice directly.
    // Here we store Decl objects in decls_; we expose a Slice<Decl> by copying out
    // the top-level order into a flat block.
    std::vector<Decl> ordered;
    ordered.reserve(toplevel_.size());
    for (DeclId id : toplevel_) {
      ordered.push_back(decls_.at(id));
    }
    tu_block_ = decl_arena_.copy_into(ordered); // persistently owned memory

    TranslationUnit tu;
    tu.decls = tu_block_;
    tu.where = {}; // optional
    return tu;
  }

  // Direct access to arenas by ID (read-only views).
  const Type& type(TypeId id) const { return types_.at(id); }
  const Expr& expr(ExprId id) const { return exprs_.at(id); }
  const Stmt& stmt(StmtId id) const { return stmts_.at(id); }
  const Decl& decl(DeclId id) const { return decls_.at(id); }

private:
  TypeId push_type(Type&& t) {
    types_.push_back(std::move(t));
    return static_cast<TypeId>(types_.size() - 1);
  }
  ExprId push_expr(Expr&& e) {
    exprs_.push_back(std::move(e));
    return static_cast<ExprId>(exprs_.size() - 1);
  }
  StmtId push_stmt(Stmt&& s) {
    stmts_.push_back(std::move(s));
    return static_cast<StmtId>(stmts_.size() - 1);
  }
  DeclId push_decl(Decl&& d) {
    decls_.push_back(std::move(d));
    return static_cast<DeclId>(decls_.size() - 1);
  }

private:
  // Owning arenas of node objects accessed by IDs.
  std::vector<Type> types_;
  std::vector<Expr> exprs_;
  std::vector<Stmt> stmts_;
  std::vector<Decl> decls_;

  // Stable storage for variable-length substructures embedded as Slice<T>.
  SegArena<Param>       params_arena_;
  SegArena<Field>       fields_arena_;
  SegArena<Enumerator>  enums_arena_;
  SegArena<Arg>         args_arena_;
  SegArena<StmtId>      stmt_list_arena_;
  SegArena<Decl>        decl_arena_;   // for TU copy-out

  // Top-level order (DeclIds).
  std::vector<DeclId>   toplevel_;

  // Interning & stable view storage.
  StringInterner        strings_;

  // Persistent TU slice (kept so returned TranslationUnit's Slice remains valid)
  Slice<Decl>           tu_block_{};
};

} // namespace see
