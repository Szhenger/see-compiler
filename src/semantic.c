#include <stdio.h>
#include <string.h>
#include "semantic.h"

typedef struct Symbol {
    char *name;
    struct Symbol *next;
} Symbol;

static Symbol *symbol_table = NULL;

static void add_symbol(const char *name) 
{
    Symbol *s = malloc(sizeof(Symbol));
    s->name = strdup(name);
    s->next = symbol_table;
    symbol_table = s;
}

static int symbol_exists(const char *name) 
{
    for (Symbol *s = symbol_table; s; s = s->next) {
        if (strcmp(s->name, name) == 0) return 1;
    }
    return 0;
}

static void clear_symbols(void) 
{
    while (symbol_table) {
        Symbol *next = symbol_table->next;
        free(symbol_table->name);
        free(symbol_table);
        symbol_table = next;
    }
}

static int analyze_call(ASTNode *node) {
    if (strcmp(node->value, "printf") != 0) {
        fprintf(stderr, "Semantic Error: Only 'printf' is supported for now\n");
        return 0;
    }

    ASTNode *arg = node->left;
    while (arg) {
        if (!analyze_expression(arg)) return 0;
        arg = arg->right;
    }

    return 1;
}

static int analyze_return(ASTNode *node) {
    if (!node->left) {
        fprintf(stderr, "Semantic Error: return without value\n");
        return 0;
    }
    return analyze_expression(node->left);
}

static int analyze_declaration(ASTNode *node) {
    if (!node->value) {
        fprintf(stderr, "Semantic Error: Declaration missing identifier\n");
        return 0;
    }
    if (symbol_exists(node->value)) {
        fprintf(stderr, "Semantic Error: Variable '%s' already declared\n", node->value);
        return 0;
    }
    add_symbol(node->value);
    return 1;
}

static int analyze_assignment(ASTNode *node) {
    if (!node->left || node->left->type != AST_IDENTIFIER) {
        fprintf(stderr, "Semantic Error: Assignment left must be identifier\n");
        return 0;
    }
    if (!symbol_exists(node->left->value)) {
        fprintf(stderr, "Semantic Error: Variable '%s' not declared\n", node->left->value);
        return 0;
    }
    return analyze_expression(node->right);
}

static int analyze_statement(ASTNode *node)
{
    switch (node->type) {
        case AST_CALL_EXPR:
            return analyze_call(node);
        case AST_RETURN_STMT:
            return analyze_return(node);
        case AST_DECLARATION:
            return analyze_declaration(node);
        case AST_ASSIGNMENT:
            return analyze_assignment(node);
        default:
            fprintf(stderr, "Semantic Error: Unknown or unsupported statement type\n");
            return 0;
    }
}

static int analyze_statement(ASTNode *node) {
    switch (node->type) {
        case AST_DECLARATION:
            return analyze_declaration(node);
        case AST_ASSIGNMENT:
            return analyze_assignment(node);
        case AST_RETURN_STMT:
            return analyze_return(node);
        case AST_CALL_EXPR:
            return analyze_call(node);
        case AST_BINARY_OP:
        case AST_IDENTIFIER:
        case AST_INTEGER_LITERAL:
        case AST_STRING_LITERAL:
            return analyze_expression(node);
        case AST_STATEMENT_LIST: {
            ASTNode *stmt = node;
            while (stmt && stmt->type == AST_STATEMENT_LIST) {
                if (!analyze_statement(stmt->left)) return 0;
                stmt = stmt->right;
            }
            if (stmt) return analyze_statement(stmt);
            return 1;
        }
        case AST_IF_STMT:
        case AST_WHILE_LOOP:
        case AST_FOR_LOOP:
            fprintf(stderr, "Semantic Error: Control flow not yet supported\n");
            return 0;
        default:
            fprintf(stderr, "Semantic Error: Unknown AST node type %d\n", node->type);
            return 0;
    }
}

static int analyze_function(ASTNode *node) {
    if (!node->left) {
        fprintf(stderr, "Semantic Error: Empty function body\n");
        return 0;
    }
    return analyze_statement(node->left);
}

SemanticResult analyze(ASTNode *root) {
    if (!root || root->type != AST_FUNCTION_DEF || strcmp(root->value, "main") != 0) {
        fprintf(stderr, "Semantic Error: Program must have a 'main' function\n");
        return SEMANTIC_ERROR;
    }

    SemanticResult result = analyze_function(root) ? SEMANTIC_OK : SEMANTIC_ERROR;
    clear_symbols();
    return result;
}
