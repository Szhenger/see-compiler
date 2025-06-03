#include "parser.h"
#include <stdlib.h>
#include <string.h>

// Utilities
static Token current_token(Parser *p) 
{
    return p->tokens[p->current];
}

static void advance(Parser *p) 
{
    if (p->current < p->length) p->current++;
}

static int match(Parser *p, TokenType type, const char *lexeme) 
{
    if (p->current >= p->length) return 0;
    Token t = current_token(p);
    if (t.type == type && (!lexeme || strcmp(t.lexeme, lexeme) == 0)) {
        advance(p);
        return 1;
    }
    return 0;
}

// Parse return statement: return 0;
static ASTNode *parse_return(Parser *p) 
{
    if (!match(p, TOKEN_KEYWORD, "return")) return NULL;

    Token val = current_token(p);
    advance(p); // skip literal

    if (!match(p, TOKEN_SYMBOL, ";")) return NULL;

    ASTNode *value = create_ast_node(AST_LITERAL, val.lexeme);
    ASTNode *ret_node = create_ast_node(AST_RETURN_STMT, NULL);
    ret_node->left = value;
    return ret_node;
}

// Parse printf("..."); → call expression
static ASTNode *parse_call(Parser *p) 
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

// Parse function body: { printf(...); return ...; }
static ASTNode *parse_function(Parser *p) 
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

ASTNode *parse(Parser *p) 
{
    return parse_function(p);
}

