#include "parser.hpp"

#include <cassert>
#include <iostream>

using token::Token;
using token::TokenCategory;
using token::TokenKind;
using namespace token;
using namespace ast;

using ASTNode = ast::Node;
using TokenCategory_t = token::TokenCategory;

Parser::Parser(const std::vector<Token>& toks)
    : tokens(toks), index(0)
{
    // Ensure there's an EOF token at the end so peek()/next() are safe.
    if (tokens.empty() || tokens.back().category != TokenCategory::EndOfFile) {
        Token eof;
        eof.category = TokenCategory::EndOfFile;
        eof.location = {1, 1};
        eof.lexeme.clear();
        tokens.push_back(eof);
    }
}

bool Parser::is_eof() const {
    return peek().category == TokenCategory::EndOfFile;
}

const Token& Parser::peek(int ahead) const {
    size_t pos = index + (ahead >= 0 ? static_cast<size_t>(ahead) : 0);
    if (pos >= tokens.size()) {
        // Return last token (EOF)
        return tokens.back();
    }
    return tokens[pos];
}

const Token& Parser::next() {
    const Token& t = peek(0);
    if (index < tokens.size()) ++index;
    return t;
}

bool Parser::match(TokenCategory kind) {
    if (peek().category == kind) {
        next();
        return true;
    }
    return false;
}

bool Parser::match_keyword(const std::string& kw) {
    const Token& t = peek();
    if (t.category == TokenCategory::Keyword && t.lexeme == kw) {
        next();
        return true;
    }
    return false;
}

bool Parser::expect(TokenCategory kind, const char* msg) {
    if (peek().category == kind) {
        next();
        return true;
    }
    // Simple diagnostic: print message and token location
    std::cerr << "Parse error: expected " << msg
              << " at " << peek().location.line << ":" << peek().location.column
              << ", got '" << peek().lexeme << "'\n";
    return false;
}

ASTNode* Parser::parse_translation_unit() {
    // Create a top-level compound statement to hold all top-level stmts/decls.
    token::SourceLocation loc{1, 1};
    CompoundStmt* root = new CompoundStmt(loc);

    while (!is_eof()) {
        // Skip any stray EOFs or unknown tokens
        if (peek().category == TokenCategory::EndOfFile) break;

        // Try parse declaration first (e.g., type-based declaration)
        ASTNode* decl = parse_declaration();
        if (decl) {
            delete decl;
            // Advance to avoid infinite loop (if parse_declaration failed to consume anything)
            if (peek().category == TokenCategory::EndOfFile) break;
            continue;
        }

        // Otherwise parse a statement
        ASTNode* stmt_node = parse_statement();
        if (stmt_node) {
            // parser expects Stmt*, but ASTNode* is accepted
            // push into root compound statements (downcast)
            if (Stmt* s = dynamic_cast<Stmt*>(stmt_node)) {
                root->statements.emplace_back(static_cast<Stmt*>(stmt_node)); // won't compile: unique_ptr required
            } else {
                delete stmt_node;
            }
        } else {
            // Failed to parse anything meaningful — consume one token to avoid infinite loop
            std::cerr << "Parser warning: skipping token '" << peek().lexeme
                      << "' at " << peek().location.line << ":" << peek().location.column << "\n";
            next();
        }
    }

    // Return the root as ASTNode*
    return root;
}

ASTNode* Parser::parse_declaration() {
    // TODO
    return nullptr;
}

ASTNode* Parser::parse_function_definition() {
    // TODO
    return nullptr;
}

ASTNode* Parser::parse_parameter_list() {
    // TODO
    return nullptr;
}

ASTNode* Parser::parse_parameter() {
    // TODO
    return nullptr;
}

// Statements

ASTNode* Parser::parse_statement() {
    // Basic dispatch: compound, if, while, for, return, or expression-statement
    const Token& t = peek();

    if (t.category == TokenCategory::Punctuation && t.lexeme == "{") {
        return parse_compound_statement();
    }

    if (t.category == TokenCategory::Keyword && t.lexeme == "if") {
        return parse_if_statement();
    }

    if (t.category == TokenCategory::Keyword && t.lexeme == "while") {
        return parse_while_statement();
    }

    if (t.category == TokenCategory::Keyword && t.lexeme == "for") {
        return parse_for_statement();
    }

    if (t.category == TokenCategory::Keyword && t.lexeme == "return") {
        return parse_return_statement();
    }

    // Expression statement fallback
    return parse_expression_statement();
}

ASTNode* Parser::parse_compound_statement() {
    // Expect '{' ... '}'
    if (!match(TokenCategory::Punctuation)) {
        return nullptr;
    }
    // For now, create an empty CompoundStmt at current location
    token::SourceLocation loc = peek().location;
    CompoundStmt* cs = new CompoundStmt(loc);

    // Consume tokens until matching '}' or EOF
    while (!is_eof()) {
        if (peek().category == TokenCategory::Punctuation && peek().lexeme == "}") {
            next(); // consume '}'
            break;
        }
        ASTNode* st = parse_statement();
        if (st) {
            // wrap into unique_ptr and push
            if (Stmt* s = dynamic_cast<Stmt*>(st)) {
                cs->statements.emplace_back(static_cast<Stmt*>(st)); // see note below
            } else {
                delete st;
            }
        } else {
            // skip token to avoid infinite loop
            next();
        }
    }
    return cs;
}

ASTNode* Parser::parse_if_statement() {
    if (!match_keyword("if")) return nullptr;
    // Expect '('
    expect(TokenCategory::Punctuation, "(");
    // For now skip until ')'
    while (!is_eof() && !(peek().category == TokenCategory::Punctuation && peek().lexeme == ")")) {
        next();
    }
    if (peek().category == TokenCategory::Punctuation && peek().lexeme == ")") next();
    // Parse then branch
    ASTNode* then_branch = parse_statement();
    // Optional else
    ASTNode* else_branch = nullptr;
    if (peek().category == TokenCategory::Keyword && peek().lexeme == "else") {
        next();
        else_branch = parse_statement();
    }

    // Build an IfStmt if possible
    token::SourceLocation loc = then_branch ? then_branch->loc : token::SourceLocation{0,0};
    IfStmt* is = new IfStmt(nullptr, then_branch ? std::unique_ptr<Stmt>(static_cast<Stmt*>(then_branch)) : nullptr,
                            else_branch ? std::unique_ptr<Stmt>(static_cast<Stmt*>(else_branch)) : nullptr, loc);
    return is;
}

ASTNode* Parser::parse_while_statement() {
    // TODO: parse condition and body
    if (!match_keyword("while")) return nullptr;
    // skip until body for skeleton
    while (!is_eof() && !(peek().category == TokenCategory::Punctuation && peek().lexeme == "{")) next();
    ASTNode* body = parse_statement();
    token::SourceLocation loc = body ? body->loc : token::SourceLocation{0,0};
    WhileStmt* ws = new WhileStmt(nullptr, body ? std::unique_ptr<Stmt>(static_cast<Stmt*>(body)) : nullptr, loc);
    return ws;
}

ASTNode* Parser::parse_for_statement() {
    // TODO: parse init; cond; step; body
    if (!match_keyword("for")) return nullptr;
    // Skip tokens until we find '{' or ';' then parse body as statement
    while (!is_eof() && !(peek().category == TokenCategory::Punctuation && peek().lexeme == "{")) next();
    ASTNode* body = parse_statement();
    token::SourceLocation loc = body ? body->loc : token::SourceLocation{0,0};
    ForStmt* fs = new ForStmt(nullptr, nullptr, nullptr, body ? std::unique_ptr<Stmt>(static_cast<Stmt*>(body)) : nullptr, loc);
    return fs;
}

ASTNode* Parser::parse_return_statement() {
    if (!match_keyword("return")) return nullptr;
    // For skeleton, skip until semicolon
    while (!is_eof() && !(peek().category == TokenCategory::Punctuation && peek().lexeme == ";")) {
        next();
    }
    if (peek().category == TokenCategory::Punctuation && peek().lexeme == ";") next();
    token::SourceLocation loc{0,0};
    ReturnStmt* rs = new ReturnStmt(nullptr, loc);
    return rs;
}

ASTNode* Parser::parse_expression_statement() {
    // Parse an expression (stub) then expect semicolon
    ASTNode* expr = parse_expression();
    if (peek().category == TokenCategory::Punctuation && peek().lexeme == ";") {
        next();
    }
    // If expr resolves to an Expr*, wrap as ExprStmt
    if (Expr* e =*
