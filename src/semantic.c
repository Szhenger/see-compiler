#include <stdio.h>
#include <string.h>
#include "ast.h"
#include "semantic.h"

// Forward Helper Declarations
static int analyze_function(ASTNode *node);
static int analyze_call(ASTNode *node);
static int analyze_return(ASTNode *node);

// === Public Function: Analyze an AST ===
SemanticResult analyze(ASTNode *root) 
{
    if (!root) {
        fprintf(stderr, "Semantic Error: NULL AST\n");
        return SEMANTIC_ERROR;
    } else if (root->type != AST_FUNCTION_DEF || strcmp(root->value, "main") != 0) {
        fprintf(stderr, "Semantic Error: Program must have a 'main' function\n");
        return SEMANTIC_ERROR;
    } else if (!analyze_function(root)) {
        return SEMANTIC_ERROR;
    } else {
        return SEMANTIC_OK;
    }
}

// === Private Helper: Analyzes a Function Node ===
static int analyze_function(ASTNode *node) 
{
    if (!node->left || !node->right) {
        fprintf(stderr, "Semantic Error: Function body incomplete\n");
        return 0;
    } else if (!analyze_call(node->left) || !analyze_return(node->right)) {
        return 0;
    } else {
        return 1;
    }
}

// === Private Helper: Analyzes a Call Expression (e.g., printf) ===
static int analyze_call(ASTNode *node) 
{
    if (node->type != AST_CALL_EXPR || strcmp(node->value, "printf") != 0) {
        fprintf(stderr, "Semantic Error: Unsupported function call - %s\n", node->value);
        return 0;
    } else if (!node->left || node->left->type != AST_STRING_LITERAL) {
        fprintf(stderr, "Semantic Error: printf must take a string literal\n");
        return 0;
    } else {
        return 1;
    }
}

// === Private Helper: Analyzes a return statement ===
static int analyze_return(ASTNode *node) 
{
    if (node->type != AST_RETURN_STMT || !node->left) {
        fprintf(stderr, "Semantic error: Malformed return statement\n");
        return 0;
    } else if (node->left->type != AST_LITERAL) {
        fprintf(stderr, "Semantic error: return must return a literal integer\n");
        return 0;
    } else {
        return 1;
    }
}
