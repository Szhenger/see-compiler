#ifndef AST_HPP
#define AST_HPP

#include <memory>
#include <string>
#include <vector>
#include "token.hpp"

namespace ast {

struct Expr;
struct Stmt;
struct Decl;

using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;
using DeclPtr = std::unique_ptr<Decl>;

struct Node {
    token::SourceLocation loc;
    virtual ~Node() = default;
};

enum class ExprKind {
    Literal,
    Identifier,
    Unary,
    Binary,
    Ternary,
    Call,
    Member,
    Cast,
    Index,
    Assign
};

struct Expr : Node {
    ExprKind kind;
    Expr(ExprKind k, const token::SourceLocation &l) : kind(k) { loc = l; }
    virtual ~Expr() = default;
};

struct LiteralExpr : Expr {
    token::Token value;
    LiteralExpr(const token::Token &t)
        : Expr(ExprKind::Literal, t.location), value(token::copy_token(t)) {}
};

struct IdentifierExpr : Expr {
    std::string name;
    IdentifierExpr(const token::Token &t)
        : Expr(ExprKind::Identifier, t.location), name(t.lexeme) {}
};

struct UnaryExpr : Expr {
    token::OperatorKind op;
    ExprPtr operand;
    UnaryExpr(token::OperatorKind oper, ExprPtr expr, const token::SourceLocation &l)
        : Expr(ExprKind::Unary, l), op(oper), operand(std::move(expr)) {}
};

struct BinaryExpr : Expr {
    token::OperatorKind op;
    ExprPtr lhs;
    ExprPtr rhs;
    BinaryExpr(token::OperatorKind oper, ExprPtr left, ExprPtr right, const token::SourceLocation &l)
        : Expr(ExprKind::Binary, l), op(oper), lhs(std::move(left)), rhs(std::move(right)) {}
};

struct TernaryExpr : Expr {
    ExprPtr cond;
    ExprPtr then_branch;
    ExprPtr else_branch;
    TernaryExpr(ExprPtr c, ExprPtr t, ExprPtr e, const token::SourceLocation &l)
        : Expr(ExprKind::Ternary, l), cond(std::move(c)), then_branch(std::move(t)), else_branch(std::move(e)) {}
};

struct CallExpr : Expr {
    ExprPtr callee;
    std::vector<ExprPtr> args;
    CallExpr(ExprPtr c, std::vector<ExprPtr> a, const token::SourceLocation &l)
        : Expr(ExprKind::Call, l), callee(std::move(c)), args(std::move(a)) {}
};

struct MemberExpr : Expr {
    ExprPtr base;
    std::string field;
    bool is_arrow;
    MemberExpr(ExprPtr b, const std::string &f, bool arrow, const token::SourceLocation &l)
        : Expr(ExprKind::Member, l), base(std::move(b)), field(f), is_arrow(arrow) {}
};

struct CastExpr : Expr {
    std::string type_name;
    ExprPtr expr;
    CastExpr(const std::string &t, ExprPtr e, const token::SourceLocation &l)
        : Expr(ExprKind::Cast, l), type_name(t), expr(std::move(e)) {}
};

struct IndexExpr : Expr {
    ExprPtr base;
    ExprPtr index;
    IndexExpr(ExprPtr b, ExprPtr i, const token::SourceLocation &l)
        : Expr(ExprKind::Index, l), base(std::move(b)), index(std::move(i)) {}
};

struct AssignExpr : Expr {
    ExprPtr target;
    token::OperatorKind op;
    ExprPtr value;
    AssignExpr(ExprPtr t, token::OperatorKind o, ExprPtr v, const token::SourceLocation &l)
        : Expr(ExprKind::Assign, l), target(std::move(t)), op(o), value(std::move(v)) {}
};

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

enum class StmtKind {
    ExprStmt,
    Compound,
    If,
    While,
    For,
    Return,
    DeclStmt
};

struct Stmt : Node {
    StmtKind kind;
    Stmt(StmtKind k, const token::SourceLocation &l) : kind(k) { loc = l; }
    virtual ~Stmt() = default;
};

struct ExprStmt : Stmt {
    ExprPtr expr;
    ExprStmt(ExprPtr e, const token::SourceLocation &l)
        : Stmt(StmtKind::ExprStmt, l), expr(std::move(e)) {}
};

struct CompoundStmt : Stmt {
    std::vector<StmtPtr> statements;
    CompoundStmt(const token::SourceLocation &l)
        : Stmt(StmtKind::Compound, l) {}
};

struct IfStmt : Stmt {
    ExprPtr cond;
    StmtPtr then_branch;
    StmtPtr else_branch;
    IfStmt(ExprPtr c, StmtPtr t, StmtPtr e, const token::SourceLocation &l)
        : Stmt(StmtKind::If, l), cond(std::move(c)), then_branch(std::move(t)), else_branch(std::move(e)) {}
};

struct WhileStmt : Stmt {
    ExprPtr cond;
    StmtPtr body;
    WhileStmt(ExprPtr c, StmtPtr b, const token::SourceLocation &l)
        : Stmt(StmtKind::While, l), cond(std::move(c)), body(std::move(b)) {}
};

struct ForStmt : Stmt {
    StmtPtr init;
    ExprPtr cond;
    ExprPtr step;
    StmtPtr body;
    ForStmt(StmtPtr i, ExprPtr c, ExprPtr s, StmtPtr b, const token::SourceLocation &l)
        : Stmt(StmtKind::For, l), init(std::move(i)), cond(std::move(c)), step(std::move(s)), body(std::move(b)) {}
};

struct ReturnStmt : Stmt {
    ExprPtr value;
    ReturnStmt(ExprPtr v, const token::SourceLocation &l)
        : Stmt(StmtKind::Return, l), value(std::move(v)) {}
};

enum class DeclKind {
    Var,
    Func,
    Param
};

struct Decl : Node {
    DeclKind kind;
    Decl(DeclKind k, const token::SourceLocation &l) : kind(k) { loc = l; }
    virtual ~Decl() = default;
};

struct VarDecl : Decl {
    std::string type_name;
    std::string name;
    ExprPtr init;
    VarDecl(const std::string &t, const std::string &n, ExprPtr i, const token::SourceLocation &l)
        : Decl(DeclKind::Var, l), type_name(t), name(n), init(std::move(i)) {}
};

struct ParamDecl : Decl {
    std::string type_name;
    std::string name;
    ParamDecl(const std::string &t, const std::string &n, const token::SourceLocation &l)
        : Decl(DeclKind::Param, l), type_name(t), name(n) {}
};

struct FuncDecl : Decl {
    std::string return_type;
    std::string name;
    std::vector<std::unique_ptr<ParamDecl>> params;
    std::unique_ptr<CompoundStmt> body;
    FuncDecl(const std::string &rt, const std::string &n, const token::SourceLocation &l)
        : Decl(DeclKind::Func, l), return_type(rt), name(n) {}
};

} // namespace ast

#endif // AST_HPP
