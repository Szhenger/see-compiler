#include <stdio.h>
#include <string.h>
#include "ast.h"
#include "semantic.h"

// Forward declarations
static int analyze_function(ASTNode *node);
static int analyze_call(ASTNode *node);
static int analyze_return(ASTNode *node);

SemanticResult analyze(ASTNode *root) {
    if (!root) {
        fprintf(stderr, "Semantic error: NULL AST\n");
        return SEMANTIC_ERROR;
    }

    if (root->type != AST_FUNCTION_DEF || strcmp(root->value, "main") != 0) {
        fprintf(stderr, "Semantic error: Program must have a 'main' function\n");
        return SEMANTIC_ERROR;
    }

    if (!analyze_function(root)) {
        return SEMANTIC_ERROR;
    }

    return SEMANTIC_OK;
}

// Analyzes a function node
static int analyze_function(ASTNode *node) {
    if (!node->left || !node->right) {
        fprintf(stderr, "Semantic error: Function body incomplete\n");
        return 0;
    }

    if (!analyze_call(node->left)) return 0;
    if (!analyze_return(node->right)) return 0;

    return 1;
}

// Analyzes a call expression (e.g., printf)
static int analyze_call(ASTNode *node) {
    if (node->type != AST_CALL_EXPR || strcmp(node->value, "printf") != 0) {
        fprintf(stderr, "Semantic error: Unsupported function call: %s\n", node->value);
        return 0;
    }

    if (!node->left || node->left->type != AST_STRING_LITERAL) {
        fprintf(stderr, "Semantic error: printf must take a string literal\n");
        return 0;
    }

    return 1;
}

// Analyzes a return statement
static int analyze_return(ASTNode *node) {
    if (node->type != AST_RETURN_STMT || !node->left) {
        fprintf(stderr, "Semantic error: Malformed return statement\n");
        return 0;
    }

    if (node->left->type != AST_LITERAL) {
        fprintf(stderr, "Semantic error: return must return a literal integer\n");
        return 0;
    }

    return 1;
}
