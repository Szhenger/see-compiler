#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "semantic.h"

// == Internal Symbol Structure ==
typedef struct Symbol {
    char *name;
    struct Symbol *next;
} Symbol;

// == Internal Symbol Table ==
static Symbol *symbol_table = NULL;

// == Private Utilties: Manage internal symbol table ==
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

// == Forward Declarations for Semantic Analysis ==
static int analyze_expression(ASTNode *node);
static int analyze_call(ASTNode *node);
static int analyze_return(ASTNode *node);
static int analyze_declaration(ASTNode *node);
static int analyze_assignment(ASTNode *node);
static int analyze_statement(ASTNode *node);
static int analyze_function(ASTNode *node);

// == Public Helper: Analyze a expression ==
static int analyze_expression(ASTNode *node) {
    switch (node->type) {
        case AST_INTEGER_LITERAL:
        case AST_STRING_LITERAL:
            return 1;
        case AST_IDENTIFIER:
            if (!symbol_exists(node->value)) {
                fprintf(stderr, "Semantic Error: Variable '%s' not declared\n", node->value);
                return 0;
            }
            return 1;
        case AST_BINARY_OP:
            return analyze_expression(node->left) && analyze_expression(node->right);
        case AST_CALL_EXPR:
            return analyze_call(node);
        default:
            fprintf(stderr, "Semantic Error: Invalid expression type %d\n", node->type);
            return 0;
    }
}

// == Private Helper: Analyze a function call semantics == 
static int analyze_call(ASTNode *node) {
    ASTNode *arg = node->left;
    while (arg) {
        if (!analyze_expression(arg)) return 0;
        arg = arg->right;
    }

    return 1;
}

// == Private Helper: Analyze the semantics of a return statement ==
static int analyze_return(ASTNode *node) 
{
    if (!node->left) {
        fprintf(stderr, "Semantic Error: return without value\n");
        return 0;
    }
    return analyze_expression(node->left);
}

// == Private Helper: Analyze the semantice of a variable declaration == 
static int analyze_declaration(ASTNode *node) 
{
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

// == Private Helper: Analyze the semantics of a variable assignment ==
static int analyze_assignment(ASTNode *node) 
{
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

// == Private Helper: Analyze the semantics of a statement == 
static int analyze_statement(ASTNode *node) 
{
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
        case AST_IF_STMT: {
            if (!analyze_expression(node->left)) {
                fprintf(stderr, "Semantic Error: Invalid condition in if-statement\n");
                return 0;
            }
            ASTNode *then_branch = node->right ? node->right->left : NULL;
            ASTNode *else_branch = node->right ? node->right->right : NULL;
            if (then_branch && !analyze_statement(then_branch)) return 0;
            if (else_branch && !analyze_statement(else_branch)) return 0;
            return 1;
        }
        case AST_WHILE_LOOP: {
            if (!analyze_expression(node->left)) {
                fprintf(stderr, "Semantic Error: Invalid condition in while-loop\n");
                return 0;
            }
            if (!analyze_statement(node->right)) return 0;
            return 1;
        }
        case AST_FOR_LOOP: {
            ASTNode *init = node->left;           
            ASTNode *test = node->right ? node->right->left : NULL;  
            ASTNode *step = node->right ? node->right->right : NULL;
        
            if (init && !analyze_statement(init)) return 0;
            if (test && !analyze_expression(test)) return 0;
            if (step && !analyze_statement(step)) return 0;
            return 1;
        }
        default:
            fprintf(stderr, "Semantic Error: Unknown AST node type %d\n", node->type);
            return 0;
    }
}

// == Private Helper: Analyze the semantics of a function body ==
static int analyze_function(ASTNode *node) 
{
    if (!node->left) {
        fprintf(stderr, "Semantic Error: Empty function body\n");
        return 0;
    }
    return analyze_statement(node->left);
}

// Public Function: Analyze the semantics of the input AST ==
SemanticResult analyze(ASTNode *root) 
{
    if (!root || root->type != AST_FUNCTION_DEF || strcmp(root->value, "main") != 0) {
        fprintf(stderr, "Semantic Error: Program must have a 'main' function\n");
        return SEMANTIC_ERROR;
    }

    SemanticResult result = analyze_function(root) ? SEMANTIC_OK : SEMANTIC_ERROR;
    clear_symbols();
    return result;
}
