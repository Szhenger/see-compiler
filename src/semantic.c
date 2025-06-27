#include <stdio.h>
#include <string.h>
#include "semantic.h"

// === Private Helper: Call expression (e.g., printf("...")) ===
static int analyze_call(ASTNode *node) 
{
    if (strcmp(node->value, "printf") != 0) {
        fprintf(stderr, "Semantic Error: Only 'printf' is supported for now\n");
        return 0;
    }
    if (!node->left || node->left->type != AST_STRING_LITERAL) {
        fprintf(stderr, "Semantic Error: printf must take a string literal\n");
        return 0;
    }
    return 1;
}

// === Private Helper: Return statement ===
static int analyze_return(ASTNode *node) 
{
    if (!node->left) {
        fprintf(stderr, "Semantic Error: return without value\n");
        return 0;
    }
    if (node->left->type != AST_LITERAL && node->left->type != AST_INTEGER_LITERAL) {
        fprintf(stderr, "Semantic Error: return must return an integer literal\n");
        return 0;
    }
    return 1;
}

// === Private Helper: Variable Declaration (e.g., int x;) ===
static int analyze_declaration(ASTNode *node)
{
    if (!node->value) {
        fprintf(stderr, "Semantic Error: Declaration missing identifier\n");
        return 0;
    }
    return 1;
}

// === Private Helper: Assignment (e.g., x = 42;) ===
static int analyze_assignment(ASTNode *node)
{
    if (!node->left || node->left->type != AST_IDENTIFIER) {
        fprintf(stderr, "Semantic Error: Assignment must assign to an identifier\n");
        return 0;
    }
    if (!node->right || node->right->type != AST_INTEGER_LITERAL) {
        fprintf(stderr, "Semantic Error: Assignment must use an integer literal\n");
        return 0;
    }
    return 1;
}

// === Private Helper: Analyze one statement ===
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

// === Private Helper: Analyze a function node ===
// Assumes function body is a statement list
static int analyze_function(ASTNode *node) 
{
    if (!node->left) {
        fprintf(stderr, "Semantic Error: Function body is empty\n");
        return 0;
    }

    ASTNode *statement = node->left;
    while (statement) {
        ASTNode *current = (statement->type == AST_STATEMENT_LIST) ? statement->left : statement;
        if (!analyze_statement(current)) return 0;

        statement = (statement->type == AST_STATEMENT_LIST) ? statement->right : NULL;
    }

    return 1;
}

// === Public Function: Analyze an AST ===
SemanticResult analyze(ASTNode *root) 
{
    if (!root) {
        fprintf(stderr, "Semantic Error: NULL AST\n");
        return SEMANTIC_ERROR;
    } 
    
    if (root->type != AST_FUNCTION_DEF || strcmp(root->value, "main") != 0) {
        fprintf(stderr, "Semantic Error: Program must have a 'main' function\n");
        return SEMANTIC_ERROR;
    }
    
    if (!analyze_function(root)) {
        return SEMANTIC_ERROR;
    }
    
    return SEMANTIC_OK;
}
