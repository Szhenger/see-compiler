#include "ast.hpp"

#include <iostream>
#include <sstream>

namespace ast {

ExprPtr make_literal_expr(const token::Token &t) {
    return std::make_unique<LiteralExpr>(t);
}

ExprPtr make_identifier_expr(const token::Token &t) {
    return std::make_unique<IdentifierExpr>(t);
}

ExprPtr make_unary_expr(token::OperatorKind op, ExprPtr operand, const token::SourceLocation &loc) {
    return std::make_unique<UnaryExpr>(op, std::move(operand), loc);
}

ExprPtr make_binary_expr(token::OperatorKind op, ExprPtr lhs, ExprPtr rhs, const token::SourceLocation &loc) {
    return std::make_unique<BinaryExpr>(op, std::move(lhs), std::move(rhs), loc);
}

ExprPtr make_ternary_expr(ExprPtr cond, ExprPtr then_branch, ExprPtr else_branch, const token::SourceLocation &loc) {
    return std::make_unique<TernaryExpr>(std::move(cond), std::move(then_branch), std::move(else_branch), loc);
}

ExprPtr make_call_expr(ExprPtr callee, std::vector<ExprPtr> args, const token::SourceLocation &loc) {
    return std::make_unique<CallExpr>(std::move(callee), std::move(args), loc);
}

ExprPtr make_member_expr(ExprPtr base, const std::string &field, bool is_arrow, const token::SourceLocation &loc) {
    return std::make_unique<MemberExpr>(std::move(base), field, is_arrow, loc);
}

ExprPtr make_cast_expr(const std::string &type_name, ExprPtr expr, const token::SourceLocation &loc) {
    return std::make_unique<CastExpr>(type_name, std::move(expr), loc);
}

ExprPtr make_index_expr(ExprPtr base, ExprPtr index, const token::SourceLocation &loc) {
    return std::make_unique<IndexExpr>(std::move(base), std::move(index), loc);
}

ExprPtr make_assign_expr(ExprPtr target, token::OperatorKind op, ExprPtr value, const token::SourceLocation &loc) {
    return std::make_unique<AssignExpr>(std::move(target), op, std::move(value), loc);
}

StmtPtr make_expr_stmt(ExprPtr expr, const token::SourceLocation &loc) {
    return std::make_unique<ExprStmt>(std::move(expr), loc);
}

StmtPtr make_compound_stmt(const token::SourceLocation &loc) {
    return std::make_unique<CompoundStmt>(loc);
}

StmtPtr make_if_stmt(ExprPtr cond, StmtPtr then_branch, StmtPtr else_branch, const token::SourceLocation &loc) {
    return std::make_unique<IfStmt>(std::move(cond), std::move(then_branch), std::move(else_branch), loc);
}

StmtPtr make_while_stmt(ExprPtr cond, StmtPtr body, const token::SourceLocation &loc) {
    return std::make_unique<WhileStmt>(std::move(cond), std::move(body), loc);
}

StmtPtr make_for_stmt(StmtPtr init, ExprPtr cond, ExprPtr step, StmtPtr body, const token::SourceLocation &loc) {
    return std::make_unique<ForStmt>(std::move(init), std::move(cond), std::move(step), std::move(body), loc);
}

StmtPtr make_return_stmt(ExprPtr value, const token::SourceLocation &loc) {
    return std::make_unique<ReturnStmt>(std::move(value), loc);
}

DeclPtr make_var_decl(const std::string &type_name, const std::string &name, ExprPtr init, const token::SourceLocation &loc) {
    return std::make_unique<VarDecl>(type_name, name, std::move(init), loc);
}

std::unique_ptr<ParamDecl> make_param(const std::string &type_name, const std::string &name, const token::SourceLocation &loc) {
    return std::make_unique<ParamDecl>(type_name, name, loc);
}

DeclPtr make_func_decl(const std::string &return_type, const std::string &name, const token::SourceLocation &loc) {
    return std::make_unique<FuncDecl>(return_type, name, loc);
}

static void indent(std::ostream &os, int n) {
    for (int i = 0; i < n; ++i) os.put(' ');
}

static void dump_node(std::ostream &os, const Node *node, int level);

// Expressions
static void dump_expr(std::ostream &os, const Expr *e, int level) {
    if (!e) {
        indent(os, level);
        os << "<null-expr>\n";
        return;
    }
    switch (e->kind) {
    case ExprKind::Literal: {
        auto *le = static_cast<const LiteralExpr*>(e);
        indent(os, level);
        os << "LiteralExpr ";
        os << "[" << le->value.location.line << ":" << le->value.location.column << "] ";
        os << le->value.lexeme << "\n";
        break;
    }
    case ExprKind::Identifier: {
        auto *ie = static_cast<const IdentifierExpr*>(e);
        indent(os, level);
        os << "IdentifierExpr " << "[" << ie->loc.line << ":" << ie->loc.column << "] " << ie->name << "\n";
        break;
    }
    case ExprKind::Unary: {
        auto *ue = static_cast<const UnaryExpr*>(e);
        indent(os, level);
        os << "UnaryExpr " << "[" << ue->loc.line << ":" << ue->loc.column << "] op=" << token::operator_to_string(ue->op) << "\n";
        dump_expr(os, ue->operand.get(), level + 2);
        break;
    }
    case ExprKind::Binary: {
        auto *be = static_cast<const BinaryExpr*>(e);
        indent(os, level);
        os << "BinaryExpr " << "[" << be->loc.line << ":" << be->loc.column << "] op=" << token::operator_to_string(be->op) << "\n";
        dump_expr(os, be->lhs.get(), level + 2);
        dump_expr(os, be->rhs.get(), level + 2);
        break;
    }
    case ExprKind::Ternary: {
        auto *te = static_cast<const TernaryExpr*>(e);
        indent(os, level);
        os << "TernaryExpr " << "[" << te->loc.line << ":" << te->loc.column << "]\n";
        dump_expr(os, te->cond.get(), level + 2);
        dump_expr(os, te->then_branch.get(), level + 2);
        dump_expr(os, te->else_branch.get(), level + 2);
        break;
    }
    case ExprKind::Call: {
        auto *ce = static_cast<const CallExpr*>(e);
        indent(os, level);
        os << "CallExpr " << "[" << ce->loc.line << ":" << ce->loc.column << "]\n";
        dump_expr(os, ce->callee.get(), level + 2);
        indent(os, level + 2); os << "Args:\n";
        for (const auto &a : ce->args) dump_expr(os, a.get(), level + 4);
        break;
    }
    case ExprKind::Member: {
        auto *me = static_cast<const MemberExpr*>(e);
        indent(os, level);
        os << "MemberExpr " << (me->is_arrow ? "->" : ".") << " " << me->field << " [" << me->loc.line << ":" << me->loc.column << "]\n";
        dump_expr(os, me->base.get(), level + 2);
        break;
    }
    case ExprKind::Cast: {
        auto *ce = static_cast<const CastExpr*>(e);
        indent(os, level);
        os << "CastExpr to " << ce->type_name << " [" << ce->loc.line << ":" << ce->loc.column << "]\n";
        dump_expr(os, ce->expr.get(), level + 2);
        break;
    }
    case ExprKind::Index: {
        auto *ie = static_cast<const IndexExpr*>(e);
        indent(os, level);
        os << "IndexExpr [" << ie->loc.line << ":" << ie->loc.column << "]\n";
        dump_expr(os, ie->base.get(), level + 2);
        dump_expr(os, ie->index.get(), level + 2);
        break;
    }
    case ExprKind::Assign: {
        auto *ae = static_cast<const AssignExpr*>(e);
        indent(os, level);
        os << "AssignExpr op=" << token::operator_to_string(ae->op) << " [" << ae->loc.line << ":" << ae->loc.column << "]\n";
        dump_expr(os, ae->target.get(), level + 2);
        dump_expr(os, ae->value.get(), level + 2);
        break;
    }
    default:
        indent(os, level);
        os << "Expr<unknown>\n";
        break;
    }
}

static void dump_stmt(std::ostream &os, const Stmt *s, int level) {
    if (!s) {
        indent(os, level);
        os << "<null-stmt>\n";
        return;
    }
    switch (s->kind) {
    case StmtKind::ExprStmt: {
        auto *es = static_cast<const ExprStmt*>(s);
        indent(os, level);
        os << "ExprStmt [" << es->loc.line << ":" << es->loc.column << "]\n";
        dump_expr(os, es->expr.get(), level + 2);
        break;
    }
    case StmtKind::Compound: {
        auto *cs = static_cast<const CompoundStmt*>(s);
        indent(os, level);
        os << "CompoundStmt [" << cs->loc.line << ":" << cs->loc.column << "]\n";
        for (const auto &st : cs->statements) dump_stmt(os, st.get(), level + 2);
        break;
    }
    case StmtKind::If: {
        auto *ifs = static_cast<const IfStmt*>(s);
        indent(os, level);
        os << "IfStmt [" << ifs->loc.line << ":" << ifs->loc.column << "]\n";
        dump_expr(os, ifs->cond.get(), level + 2);
        indent(os, level + 2); os << "Then:\n";
        dump_stmt(os, ifs->then_branch.get(), level + 4);
        if (ifs->else_branch) {
            indent(os, level + 2); os << "Else:\n";
            dump_stmt(os, ifs->else_branch.get(), level + 4);
        }
        break;
    }
    case StmtKind::While: {
        auto *ws = static_cast<const WhileStmt*>(s);
        indent(os, level);
        os << "WhileStmt [" << ws->loc.line << ":" << ws->loc.column << "]\n";
        dump_expr(os, ws->cond.get(), level + 2);
        dump_stmt(os, ws->body.get(), level + 2);
        break;
    }
    case StmtKind::For: {
        auto *fs = static_cast<const ForStmt*>(s);
        indent(os, level);
        os << "ForStmt [" << fs->loc.line << ":" << fs->loc.column << "]\n";
        indent(os, level + 2); os << "Init:\n";
        if (fs->init) dump_stmt(os, fs->init.get(), level + 4);
        indent(os, level + 2); os << "Cond:\n";
        if (fs->cond) dump_expr(os, fs->cond.get(), level + 4);
        indent(os, level + 2); os << "Step:\n";
        if (fs->step) dump_expr(os, fs->step.get(), level + 4);
        indent(os, level + 2); os << "Body:\n";
        dump_stmt(os, fs->body.get(), level + 4);
        break;
    }
    case StmtKind::Return: {
        auto *rs = static_cast<const ReturnStmt*>(s);
        indent(os, level);
        os << "ReturnStmt [" << rs->loc.line << ":" << rs->loc.column << "]\n";
        if (rs->value) dump_expr(os, rs->value.get(), level + 2);
        break;
    }
    case StmtKind::DeclStmt: {
        indent(os, level);
        os << "DeclStmt (not implemented detail) [" << s->loc.line << ":" << s->loc.column << "]\n";
        break;
    }
    default:
        indent(os, level);
        os << "Stmt<unknown>\n";
        break;
    }
}

// Declarations
static void dump_decl(std::ostream &os, const Decl *d, int level) {
    if (!d) {
        indent(os, level);
        os << "<null-decl>\n";
        return;
    }
    switch (d->kind) {
    case DeclKind::Var: {
        auto *vd = static_cast<const VarDecl*>(d);
        indent(os, level);
        os << "VarDecl " << vd->name << " : " << vd->type_name << " [" << vd->loc.line << ":" << vd->loc.column << "]\n";
        if (vd->init) {
            indent(os, level + 2);
            os << "Init:\n";
            dump_expr(os, vd->init.get(), level + 4);
        }
        break;
    }
    case DeclKind::Param: {
        auto *pd = static_cast<const ParamDecl*>(d);
        indent(os, level);
        os << "ParamDecl " << pd->name << " : " << pd->type_name << " [" << pd->loc.line << ":" << pd->loc.column << "]\n";
        break;
    }
    case DeclKind::Func: {
        auto *fd = static_cast<const FuncDecl*>(d);
        indent(os, level);
        os << "FuncDecl " << fd->name << " -> " << fd->return_type << " [" << fd->loc.line << ":" << fd->loc.column << "]\n";
        indent(os, level + 2); os << "Params:\n";
        for (const auto &p : fd->params) {
            dump_decl(os, p.get(), level + 4);
        }
        if (fd->body) {
            indent(os, level + 2); os << "Body:\n";
            dump_stmt(os, fd->body.get(), level + 4);
        }
        break;
    }
    default:
        indent(os, level);
        os << "Decl<unknown>\n";
        break;
    }
}

static void dump_node(std::ostream &os, const Node *node, int level) {
    if (!node) {
        indent(os, level);
        os << "<null-node>\n";
        return;
    }
    // dispatch based on RTTI-free kinds (we have base classes with kind fields)
    // try casting via dynamic_cast isn't necessary; instead test via known base types via pointers
    // We use dynamic_cast here for simplicity and readability; if RTTI is disabled, this can be changed.
    if (const Expr *e = dynamic_cast<const Expr*>(node)) {
        dump_expr(os, e, level);
        return;
    }
    if (const Stmt *s = dynamic_cast<const Stmt*>(node)) {
        dump_stmt(os, s, level);
        return;
    }
    if (const Decl *d = dynamic_cast<const Decl*>(node)) {
        dump_decl(os, d, level);
        return;
    }
    // fallback
    indent(os, level);
    os << "Node<unknown>\n";
}

void dump_ast(const Node *root) {
    dump_node(std::cout, root, 0);
}

void print_ast(const Node *root) {
    dump_ast(root);
}

} // namespace ast
