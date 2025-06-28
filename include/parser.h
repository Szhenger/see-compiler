#ifndef PARSER_H
#define PARSER_H

#include "ast.h"
#include "token.h"

typedef struct {
    Token *tokens; 
    int current; 
    int length;
} Parser;

Parser *init_parser(Token *t, int *count);
void free_parser(Parser *p);

ASTNode *parse(Parser *p);

#endif

