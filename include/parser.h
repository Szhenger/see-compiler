#ifndef PARSER_H
#define PARSER_H

#include "ast.h"
#include "token.h"

// === Parser Structure ===
typedef struct {
    Token *tokens;  // Token stream
    int current;    // Index of current token
    int length;     // Total number of tokens
} Parser;

// === Parser Lifecycle ===
Parser *init_parser(Token *t, int *count);
void free_parser(Parser *p);

// === Entry Point ===
ASTNode *parse(Parser *p);

#endif // PARSER_H

