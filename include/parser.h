#ifndef PARSER_H
#define PARSER_H

#include "ast.h"
#include "token.h"

// Define the structure of a parser variable
typedef struct {
    Token *tokens;
    int current;
    int length;
} Parser;

// Constructs a parser from a token stream
Parser *init_parser(Token *t, int *count);

// Parser Utilities: Token Inspection and Advancement
Token current_token(Parser *p);
void advance(Parser *p);
int match(Parser *p, TokenType type, const char *lexeme);

// Parse the return statement of the form: return <int>;
// Assumes 'return' has been matched
ASTNode *parse_return(Parser *p);

// Parse a function call expression (currently only supports printf)
ASTNode *parse_call(Parser *p);

// Parse a function body enclosed in braces: { printf(...); return ...; }
// Assumes a flat sequence of statements inide main
ASTNode *parse_function(Parser *p);

// Entry Point: Parse the main procedure including the signature, body, and return
ASTNode *parse(Parser *p);

// Destory parser and free dynamically allocated resources
void free_parser(Parser *p);

#endif
