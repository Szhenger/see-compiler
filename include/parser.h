#ifndef PARSER_H
#define PARSER_H

#include "ast.h"
#include "token.h"

// === Parser State ===
typedef struct {
    Token *tokens;
    int current;
    int length;
} Parser;

// === Initialization & Cleanup ===
Parser *init_parser(Token *tokens, int *count);
void free_parser(Parser *p);

// === Token Inspection & Movement ===
Token current_token(Parser *p);
void advance(Parser *p);
int match(Parser *p, TokenType type, const char *lexeme);
int match_type(Parser *p, TokenType type);

// === Statement Parsing ===
// Parses a return statement: return <expr>;
ASTNode *parse_return(Parser *p);

// Parses a function call (e.g., printf(...));
ASTNode *parse_call(Parser *p);

// Parses an expression — currently only supports literals and identifiers
ASTNode *parse_expression(Parser *p);

// Parses a declaration like: int x;
ASTNode *parse_declaration(Parser *p);

// Parses a single statement: return, call, declaration, etc.
ASTNode *parse_statement(Parser *p);

// Parses a block: { <stmt>; <stmt>; ... }
ASTNode *parse_block(Parser *p);

// Parses a function definition: int main(...) { ... }
ASTNode *parse_function(Parser *p);

// === Entry Point ===
ASTNode *parse(Parser *p);

#endif // PARSER_H

