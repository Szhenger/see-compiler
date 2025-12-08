#ifndef PARSER_HPP
#define PARSER_HPP

#include <vector>
#include "token.hpp"
#include "ast.hpp"
#include "lexer.hpp"

struct Parser
{
    // --- Input ---
    std::vector<Token> tokens;
    size_t index;             // current token index

    // --- Construction ---
    explicit Parser(const std::vector<Token>& toks);

    // --- Core functionality ---
    ASTNode* parse_translation_unit();     // top-level entry for C/C++

    // --- Utility ---
    bool is_eof() const;
    const Token& peek(int ahead = 0) const;
    const Token& next();
    bool match(TokenKind kind);
    bool match_keyword(const std::string& kw);
    bool expect(TokenKind kind, const char* msg);

private:
    // --- Declarations ---
    ASTNode* parse_declaration();
    ASTNode* parse_function_definition();
    ASTNode* parse_parameter_list();
    ASTNode* parse_parameter();

    // --- Statements ---
    ASTNode* parse_statement();
    ASTNode* parse_compound_statement();
    ASTNode* parse_if_statement();
    ASTNode* parse_while_statement();
    ASTNode* parse_for_statement();
    ASTNode* parse_return_statement();
    ASTNode* parse_expression_statement();

    // --- Expressions (precedence climbing or recursive descent) ---
    ASTNode* parse_expression();
    ASTNode* parse_assignment_expression();
    ASTNode* parse_binary_expression(int min_precedence);
    ASTNode* parse_unary_expression();
    ASTNode* parse_primary_expression();

    // --- Types ---
    ASTNode* parse_type_specifier();       // int, float, vector<int>, struct X
    ASTNode* parse_pointer_suffix(ASTNode* base_type);    // *, **

    // --- Helpers ---
    bool is_type_start(const Token& t) const;
};

#endif // PARSER_HPP

