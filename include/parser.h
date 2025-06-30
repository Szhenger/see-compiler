#ifndef PARSER_H
#define PARSER_H

#include "ast.h"
#include "token.h"

// Defines the structure of parser 
typedef struct {
    Token *tokens; // Token Stream Field
    int current;   // Current Token Index Field
    int length;    // Length of Stream Field
} Parser;

// Initializes a parser from a input token stream 
Parser *init_parser(Token *ts, int *count);

// Frees the input parser
void free_parser(Parser *p);

// Analyzes the syntax of token stream
ASTNode *parse(Parser *p);

#endif

