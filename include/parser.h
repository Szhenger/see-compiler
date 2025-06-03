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

// Construct a new parser instance
Parser* init_parser(Token* tokens, int length);

// Get the associated AST for parser variable
ASTNode* parse(Parser* parser);  // Entry point

// Destroy the input parser instance
void free_parser(Parser* parser);

#endif
