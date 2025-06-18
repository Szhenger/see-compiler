#include <stdlib.h>
#include <string.h>
#include "parser.h"

/*
 * Parser Strategy:
 * This is a hand-written, recursive-descent parser for a minimal subset of C.
 * It assumes the source program contains a single function: `int main(void) { ... }`
 *
 * The parser operates over a flat stream of tokens and builds an abstract syntax tree (AST).
 * It currently supports:
 * - Function calls: printf("...")
 * - Return statements: return <integer>;
 * - A flat sequence of statements in the function body (no nesting or control flow).
 *
 * The parser expects syntactically valid input; it performs limited error recovery.
 *
 * Grammar (subset):
 *     program         ::= function
 *     function        ::= 'int' 'main' '(' 'void' ')' '{' statement+ '}'
 *     statement       ::= call_stmt | return_stmt
 *     call_stmt       ::= IDENTIFIER '(' STRING_LITERAL ')' ';'
 *     return_stmt     ::= 'return' INTEGER_LITERAL ';'
 */

// === Public Function: Construct parser from a token stream and count integer ===
Parser *init_parser(Token *t, int *count) 
{
    Parser *p = malloc(sizeof(Parser));
    p->tokens = t;
    p->length = *count;
    p->current = 0;
    return p;
}

// === Private Helper: Utilities for Token Inspection and Advancement ===
Token current_token(Parser *p) 
{
    return p->tokens[p->current];
}
void advance(Parser *p) 
{
    if (p->current < p->length) p->current++;
}
int match(Parser *p, TokenType type, const char *lexeme) 
{
    if (p->current >= p->length) return 0;
    Token t = current_token(p);
    if (t.type == type && (!lexeme || strcmp(t.lexeme, lexeme) == 0)) {
        advance(p);
        return 1;
    }
    return 0;
}

// === Public Helper: Parse return statement: return 0; ===
ASTNode *parse_return(Parser *p) 
{
    if (!match(p, TOKEN_KEYWORD, "return")) return NULL;

    Token val = current_token(p);
    advance(p);

    if (!match(p, TOKEN_SYMBOL, ";")) return NULL;

    ASTNode *value = create_ast_node(AST_LITERAL, val.lexeme);
    ASTNode *ret_node = create_ast_node(AST_RETURN_STMT, NULL);
    ret_node->left = value;
    return ret_node;
}

// === Public Helper: Parse printf("..."); ===
ASTNode *parse_call(Parser *p) 
{
    Token func = current_token(p);
    if (func.type != TOKEN_IDENTIFIER) return NULL;
    advance(p);

    if (!match(p, TOKEN_SYMBOL, "(")) return NULL;

    Token arg = current_token(p);
    advance(p);

    if (!match(p, TOKEN_SYMBOL, ")")) return NULL;
    if (!match(p, TOKEN_SYMBOL, ";")) return NULL;

    ASTNode *arg_node = create_ast_node(AST_STRING_LITERAL, arg.lexeme);
    ASTNode *call_node = create_ast_node(AST_CALL_EXPR, func.lexeme);
    call_node->left = arg_node;
    return call_node;
}

// === Public Helper: Parse function body: { printf(...); return ...; } ===
ASTNode *parse_function(Parser *p) 
{
    while (p->current < p->length && !match(p, TOKEN_SYMBOL, "{")) {
        advance(p);
    }

    ASTNode *call = parse_call(p);
    ASTNode *ret = parse_return(p);

    if (!match(p, TOKEN_SYMBOL, "}")) return NULL;

    ASTNode *func = create_ast_node(AST_FUNCTION_DEF, "main");
    func->left = call;
    func->right = ret;
    return func;
}

// === Public Function: Parse main function ===
ASTNode *parse(Parser *p) 
{
    return parse_function(p);
}

// === Public Function: Destroy parser and free resources ===
void free_parser(Parser *p)
{
    if (!p) return;
    free(p);
}

