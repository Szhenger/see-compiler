#include <stdlib.h>
#include <string.h>
#include "parser.h"

/// === Public Function: Initialize a parser with token stream and count ===
Parser *init_parser(Token *t, int *count) {
    Parser *p = malloc(sizeof(Parser));
    p->tokens = t;
    p->length = *count;
    p->current = 0;
    return p;
}

// === Private Helper: Get the current token from parser ===
static Token current_token(Parser *p) {
    return p->tokens[p->current];
}

// === Private Helper: Advance parser to next token ===
static void advance(Parser *p) {
    if (p->current < p->length) p->current++;
}

// === Private Helper: Match current token against expected type and lexeme ===
static int match(Parser *p, TokenCategory category, const char *lexeme) {
    if (p->current >= p->length) return 0;
    Token t = current_token(p);
    if (t.category == category && (!lexeme || strcmp(t.lexeme, lexeme) == 0)) {
        advance(p);
        return 1;
    }
    return 0;
}

// === Private Grammar: return_stmt ::= 'return' INTEGER_LITERAL ';' ===
static ASTNode *parse_return(Parser *p) {
    if (!match(p, TOKEN_KEYWORD, "return")) return NULL;
    Token val = current_token(p);
    advance(p);
    if (!match(p, TOKEN_SEMICOLON, ";")) return NULL;

    ASTNode *value = create_ast_node(AST_LITERAL, val.lexeme);
    ASTNode *ret_node = create_ast_node(AST_RETURN_STMT, NULL);
    ret_node->left = value;
    return ret_node;
}

// === Private Grammar: call_stmt ::= IDENTIFIER '(' STRING_LITERAL ')' ';' ===
static ASTNode *parse_call(Parser *p) {
    Token func = current_token(p);
    if (func.category != TOKEN_IDENTIFIER) return NULL;
    advance(p);

    if (!match(p, TOKEN_LPAREN, "(")) return NULL;
    Token arg = current_token(p);
    advance(p);
    if (!match(p, TOKEN_RPAREN, ")")) return NULL;
    if (!match(p, TOKEN_SEMICOLON, ";")) return NULL;

    ASTNode *arg_node = create_ast_node(AST_STRING_LITERAL, arg.lexeme);
    ASTNode *call_node = create_ast_node(AST_CALL_EXPR, func.lexeme);
    call_node->left = arg_node;
    return call_node;
}

// === Private Grammar: decl_stmt ::= 'int' IDENTIFIER ';' ===
static ASTNode *parse_declaration(Parser *p) {
    if (!match(p, TOKEN_KEYWORD, "int")) return NULL;
    Token ident = current_token(p);
    if (ident.category != TOKEN_IDENTIFIER) return NULL;
    advance(p);
    if (!match(p, TOKEN_SEMICOLON, ";")) return NULL;

    return create_ast_node(AST_DECLARATION, ident.lexeme);
}

// === Private Grammar: assign_stmt ::= IDENTIFIER '=' INTEGER_LITERAL ';' ===
static ASTNode *parse_assignment(Parser *p) {
    Token ident = current_token(p);
    if (ident.category != TOKEN_IDENTIFIER) return NULL;
    advance(p);
    if (!match(p, TOKEN_ASSIGN, "=")) return NULL;

    Token val = current_token(p);
    if (val.category != TOKEN_INTEGER_LITERAL) return NULL;
    advance(p);
    if (!match(p, TOKEN_SEMICOLON, ";")) return NULL;

    ASTNode *lhs = create_ast_node(AST_IDENTIFIER, ident.lexeme);
    ASTNode *rhs = create_ast_node(AST_INTEGER_LITERAL, val.lexeme);
    ASTNode *assign = create_ast_node(AST_ASSIGNMENT, NULL);
    assign->left = lhs;
    assign->right = rhs;
    return assign;
}

// === Private Grammar: statement ::= decl_stmt | assign_stmt | call_stmt | return_stmt ===
static ASTNode *parse_statement(Parser *p) {
    int saved = p->current;
    ASTNode *stmt = NULL;

    if ((stmt = parse_declaration(p)) != NULL) return stmt;
    p->current = saved;
    if ((stmt = parse_assignment(p)) != NULL) return stmt;
    p->current = saved;
    if ((stmt = parse_call(p)) != NULL) return stmt;
    p->current = saved;
    if ((stmt = parse_return(p)) != NULL) return stmt;

    return NULL;
}

// === Private Grammar: statement_list ::= statement statement_list | ε ===
static ASTNode *parse_statement_list(Parser *p) {
    ASTNode *head = parse_statement(p);
    if (!head) return NULL;

    ASTNode *tail = parse_statement_list(p);
    if (!tail) return head;

    ASTNode *list = create_ast_node(AST_STATEMENT_LIST, NULL);
    list->left = head;
    list->right = tail;
    return list;
}

// === Private Grammar: function ::= 'int' IDENTIFIER '(' 'void' ')' '{' statement_list '}' ===
static ASTNode *parse_function(Parser *p) {
    // Skip to opening brace (used when parsing full C function signature is unnecessary)
    while (p->current < p->length && !match(p, TOKEN_LBRACE, "{")) {
        advance(p);
    }

    ASTNode *body = parse_statement_list(p);
    if (!match(p, TOKEN_RBRACE, "}")) return NULL;

    ASTNode *func = create_ast_node(AST_FUNCTION_DEF, "main");
    func->left = body;
    return func;
}

// === Public Function: Entry point ===
ASTNode *parse(Parser *p) {
    return parse_function(p);
}

// === Public Function: Free parser memory ===
void free_parser(Parser *p) {
    if (!p) return;
    free(p);
}



