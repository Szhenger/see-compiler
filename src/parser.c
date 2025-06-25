#include <stdlib.h>
#include <string.h>
#include "parser.h"

// === Initialization ===
Parser *init_parser(Token *t, int *count) {
    Parser *p = malloc(sizeof(Parser));
    p->tokens = t;
    p->length = *count;
    p->current = 0;
    return p;
}

// === Token Helpers ===
Token current_token(Parser *p) {
    return p->tokens[p->current];
}

void advance(Parser *p) {
    if (p->current < p->length) p->current++;
}

int match(Parser *p, TokenType type, const char *lexeme) {
    if (p->current >= p->length) return 0;
    Token t = current_token(p);
    if (t.category == type && (!lexeme || strcmp(t.lexeme, lexeme) == 0)) {
        advance(p);
        return 1;
    }
    return 0;
}

int match_type(Parser *p, TokenType type) {
    if (p->current >= p->length) return 0;
    if (current_token(p).type == type) {
        advance(p);
        return 1;
    }
    return 0;
}

// === Expression Parser (simplified) ===
ASTNode *parse_expression(Parser *p) {
    Token t = current_token(p);
    if (t.category == TOKEN_INTEGER_LITERAL || t.category == TOKEN_IDENTIFIER) {
        advance(p);
        return create_ast_node(AST_LITERAL, t.lexeme);
    }
    return NULL;
}

// === Statement Parsers ===
ASTNode *parse_return(Parser *p) {
    if (!match(p, TOKEN_KEYWORD, "return")) return NULL;

    ASTNode *val = parse_expression(p);
    if (!match(p, TOKEN_SYMBOL, ";")) return NULL;

    ASTNode *ret = create_ast_node(AST_RETURN_STMT, NULL);
    ret->left = val;
    return ret;
}

ASTNode *parse_call(Parser *p) {
    Token fn = current_token(p);
    if (fn.category != TOKEN_IDENTIFIER) return NULL;
    advance(p);

    if (!match(p, TOKEN_SYMBOL, "(")) return NULL;

    Token arg = current_token(p);
    if (arg.category != TOKEN_STRING_LITERAL) return NULL;
    advance(p);

    if (!match(p, TOKEN_SYMBOL, ")")) return NULL;
    if (!match(p, TOKEN_SYMBOL, ";")) return NULL;

    ASTNode *arg_node = create_ast_node(AST_STRING_LITERAL, arg.lexeme);
    ASTNode *call = create_ast_node(AST_CALL_EXPR, fn.lexeme);
    call->left = arg_node;
    return call;
}

ASTNode *parse_declaration(Parser *p) {
    if (!match(p, TOKEN_KEYWORD, "int")) return NULL;

    Token id = current_token(p);
    if (id.category != TOKEN_IDENTIFIER) return NULL;
    advance(p);

    // Optional initialization (e.g., int x = 5;)
    if (match(p, TOKEN_SYMBOL, "=")) {
        ASTNode *rhs = parse_expression(p);
        if (!match(p, TOKEN_SYMBOL, ";")) return NULL;
        // You can define a new AST_ASSIGN node if needed
        return rhs; // TEMP: just return RHS for now
    }

    if (!match(p, TOKEN_SYMBOL, ";")) return NULL;
    return create_ast_node(AST_LITERAL, id.lexeme); // TEMP: represent declaration as literal
}

ASTNode *parse_statement(Parser *p) {
    Token t = current_token(p);

    if (t.category == TOKEN_KEYWORD && strcmp(t.lexeme, "return") == 0)
        return parse_return(p);
    if (t.category == TOKEN_KEYWORD && strcmp(t.lexeme, "int") == 0)
        return parse_declaration(p);
    if (t.category == TOKEN_IDENTIFIER)
        return parse_call(p);

    return NULL;
}

// === Block Parser: Parses a flat block of statements inside { ... } ===
ASTNode *parse_block(Parser *p) {
    if (!match(p, TOKEN_SYMBOL, "{")) return NULL;

    ASTNode *head = NULL;
    ASTNode *tail = NULL;

    while (!match(p, TOKEN_SYMBOL, "}")) {
        ASTNode *stmt = parse_statement(p);
        if (!stmt) break;

        if (!head) {
            head = stmt;
            tail = stmt;
        } else {
            tail->right = stmt;
            tail = stmt;
        }
    }

    return head;
}

// === Function Parser ===
ASTNode *parse_function(Parser *p) {
    if (!match(p, TOKEN_KEYWORD, "int")) return NULL;
    if (!match(p, TOKEN_IDENTIFIER, "main")) return NULL;
    if (!match(p, TOKEN_SYMBOL, "(")) return NULL;
    if (!match(p, TOKEN_KEYWORD, "void")) return NULL;
    if (!match(p, TOKEN_SYMBOL, ")")) return NULL;

    ASTNode *body = parse_block(p);
    ASTNode *func = create_ast_node(AST_FUNCTION_DEF, "main");
    func->left = body;
    return func;
}

// === Entry Point ===
ASTNode *parse(Parser *p) {
    return parse_function(p);
}

// === Cleanup ===
void free_parser(Parser *p) {
    if (!p) return;
    free(p);
}


