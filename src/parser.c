#include "parser.h"
#include <stdlib.h>
#include <string.h>

static Token current_token(Parser* p) 
{
    return p->tokens[p->current];
}

static void advance(Parser* p) 
{
    if (p->current < p->length) p->current++;
}

ASTNode* parse_return(Parser* p) 
{
    // Assuming: return LITERAL ;
    advance(p); // skip 'return'
    Token val = current_token(p);
    advance(p); // skip literal
    advance(p); // skip ';'

    ASTNode* value = create_ast_node(AST_LITERAL, val.lexeme);
    ASTNode* ret_node = create_ast_node(AST_RETURN_STMT, NULL);
    ret_node->left = value;
    return ret_node;
}

ASTNode* parse(Parser* p) 
{
    // Extremely basic: parse `int main() { printf(...); return ...; }`
    // Skip 'int main ( ) {'
    while (p->current < p->length && strcmp(current_token(p).lexeme, "{") != 0) {
        advance(p);
    }
    advance(p); // skip '{'

    // Parse call
    Token call = current_token(p);
    ASTNode* call_node = create_ast_node(AST_CALL_EXPR, call.lexeme);
    advance(p); // skip 'printf'
    advance(p); // skip '('
    Token str = current_token(p); // string literal
    ASTNode* arg_node = create_ast_node(AST_STRING_LITERAL, str.lexeme);
    call_node->left = arg_node;
    advance(p); // skip string
    advance(p); // skip ')'
    advance(p); // skip ';'

    // Parse return
    ASTNode* ret_node = parse_return(p);

    // Wrap it all in a function node
    ASTNode* func_node = create_ast_node(AST_FUNCTION_DEF, "main");
    func_node->left = call_node;
    func_node->right = ret_node;

    return func_node;
}
