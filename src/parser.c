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

// Private Helper: Return precedence of operator token; higher means tighter binding
static int get_precedence(TokenCategory category) {
    switch (category) {
        case TOKEN_OR:             return 1;  // ||
        case TOKEN_AND:            return 2;  // &&
        case TOKEN_EQUAL:
        case TOKEN_NOT_EQUAL:      return 3;  // ==, !=
        case TOKEN_LESS:
        case TOKEN_GREATER:
        case TOKEN_LESS_EQUAL:
        case TOKEN_GREATER_EQUAL:  return 4;  // <, >, <=, >=
        case TOKEN_PLUS:
        case TOKEN_MINUS:          return 5;  // +, -
        case TOKEN_STAR:
        case TOKEN_SLASH:
        case TOKEN_PERCENT:        return 6;  // *, /, %
        default:                   return 0;  // Not an operator
    }
}

static ASTNode *parse_if(Parser *p) {
    if (!match(p, TOKEN_KEYWORD, "if")) return NULL;
    if (!match(p, TOKEN_LPAREN, "(")) return NULL;

    ASTNode *condition = parse_expression(p);
    if (!condition || !match(p, TOKEN_RPAREN, ")")) return NULL;

    ASTNode *then_branch = parse_statement(p);
    if (!then_branch) return NULL;

    ASTNode *else_branch = NULL;
    if (match(p, TOKEN_KEYWORD, "else")) {
        else_branch = parse_statement(p);
        if (!else_branch) return NULL;
    }

    ASTNode *if_node = create_ast_node(AST_IF_STMT, NULL);
    if_node->left = condition;
    if_node->right = create_ast_node(AST_STATEMENT_LIST, NULL);
    if_node->right->left = then_branch;
    if_node->right->right = else_branch;
    return if_node;
}

static ASTNode *parse_while(Parser *p) {
    if (!match(p, TOKEN_KEYWORD, "while")) return NULL;
    if (!match(p, TOKEN_LPAREN, "(")) return NULL;

    ASTNode *condition = parse_expression(p);
    if (!condition || !match(p, TOKEN_RPAREN, ")")) return NULL;

    ASTNode *body = parse_statement(p);
    if (!body) return NULL;

    ASTNode *while_node = create_ast_node(AST_WHILE_LOOP, NULL);
    while_node->left = condition;
    while_node->right = body;
    return while_node;
}

static ASTNode *parse_expression_statement(Parser *p) {
    ASTNode *expr = parse_expression(p);
    if (!expr || !match(p, TOKEN_SEMICOLON, ";")) return NULL;

    ASTNode *stmt = create_ast_node(AST_EXPRESSION_STMT, NULL);
    stmt->left = expr;
    return stmt;
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

    ASTNode *call_node = create_ast_node(AST_CALL_EXPR, func.lexeme);
    ASTNode *last_arg = NULL;
    ASTNode *args_head = NULL;

    if (!match(p, TOKEN_RPAREN, ")")) {
        while (1) {
            ASTNode *arg = parse_expression(p);
            if (!arg) return NULL;

            if (!args_head) {
                args_head = arg;
            } else {
                last_arg->right = arg;
            }
            last_arg = arg;

            if (match(p, TOKEN_RPAREN, ")")) break;
            if (!match(p, TOKEN_COMMA, ",")) return NULL;
        }
    }

    call_node->left = args_head;

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

// === Private Helper ===
static ASTNode *parse_primary(Parser *p) {
    Token t = current_token(p);

    if (t.category == TOKEN_INTEGER_LITERAL) {
        advance(p);
        return create_ast_node(AST_INTEGER_LITERAL, t.lexeme);
    }

    if (t.category == TOKEN_STRING_LITERAL) {
        advance(p);
        return create_ast_node(AST_STRING_LITERAL, t.lexeme);
    }

    if (t.category == TOKEN_IDENTIFIER) {
        advance(p);
        if (match(p, TOKEN_LPAREN, "(")) {
            p->current--;
            return parse_call(p);
        }
        return create_ast_node(AST_IDENTIFIER, t.lexeme);
    }

    if (match(p, TOKEN_LPAREN, "(")) {
        ASTNode *expr = parse_expression(p);
        match(p, TOKEN_RPAREN, ")");
        return expr;
    }

    return NULL;
}

// === Private Helper: PEMDAS ===
static ASTNode *parse_expression_with_precedence(Parser *p, int min_precedence) {
    ASTNode *left = parse_primary(p);
    if (!left) return NULL;

    while (true) {
        Token t = current_token(p);
        int prec = get_precedence(t.category);
        if (prec == 0 || prec < min_precedence) break;

        advance(p);  // consume operator token

        // Parse right side with higher precedence to handle associativity
        ASTNode *right = parse_expression_with_precedence(p, prec + 1);
        if (!right) return NULL;

        ASTNode *binop = create_ast_node(AST_BINARY_OP, t.lexeme);
        binop->left = left;
        binop->right = right;
        left = binop;
    }

    return left;
}

// === Private Grammer: expres_stmt ::= .... ===
static ASTNode *parse_expression(Parser *p) {
    return parse_expression_with_precedence(p, 1);
}

// === Private Grammar: assign_stmt ::= IDENTIFIER '=' INTEGER_LITERAL ';' ===
// assign_stmt ::= IDENTIFIER '=' expression ';'
static ASTNode *parse_assignment(Parser *p) {
    Token ident = current_token(p);
    if (ident.category != TOKEN_IDENTIFIER) return NULL;
    advance(p);

    if (!match(p, TOKEN_ASSIGN, "=")) return NULL;

    ASTNode *rhs = parse_expression(p);
    if (!match(p, TOKEN_SEMICOLON, ";")) return NULL;

    ASTNode *lhs = create_ast_node(AST_IDENTIFIER, ident.lexeme);
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
    if ((stmt = parse_if(p)) != NULL) return stmt;
    p->current = saved;
    if ((stmt = parse_while(p)) != NULL) return stmt;
    p->current = saved;
    if ((stmt = parse_call(p)) != NULL) {
        if (!match(p, TOKEN_SEMICOLON, ";")) return NULL;
        return stmt;
    }
    p->current = saved;
    if ((stmt = parse_return(p)) != NULL) return stmt;
    p->current = saved;
    if ((stmt = parse_expression_statement(p)) != NULL) return stmt;

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
