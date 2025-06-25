#ifndef PARSER_H
#define PARSER_H

#include "ast.h"
#include "token.h"

// Represents the state of the parser
typedef struct {
    Token *tokens;  // Token stream
    int current;    // Index of current token
    int length;     // Total number of tokens
} Parser;

// Parser lifecycle
Parser *init_parser(Token *t, int *count);
void free_parser(Parser *p);

// Token inspection utilities
Token current_token(Parser *p);
void advance(Parser *p);
int match(Parser *p, TokenType type, const char *lexeme);

// Statement-level parsing functions
ASTNode *parse_return(Parser *p);
ASTNode *parse_call(Parser *p);
ASTNode *parse_declaration(Parser *p);
ASTNode *parse_assignment(Parser *p);
ASTNode *parse_statement(Parser *p);
ASTNode *parse_statement_list(Parser *p);

// Function-level parsing
ASTNode *parse_function(Parser *p);

// Entry point
ASTNode *parse(Parser *p);

#endif // PARSER_H

