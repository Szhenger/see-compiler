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

// Construct parser instance
Parser *init_parser(Token *t, int *count);

// Utilities
Token current_token(Parser *p);
void advance(Parser *p);
int match(Parser *p, TokenType type, const char *lexeme);

// Parse return statement: return 0;
ASTNode *parse_return(Parser *p);

// Parse printf("...");
ASTNode *parse_call(Parser *p);

// Parse function body: { printf(...); return ...; }
ASTNode *parse_function(Parser *p);

// Parse main procedure
ASTNode *parse(Parser *p);

// Destory parser instance
void free_parser(Parser *p);
