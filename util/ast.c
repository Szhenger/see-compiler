#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"

// === Public Function: Create a new AST node with given type and value ===
ASTNode *create_ast_node(ASTNodeType type, const char *value) 
{
    ASTNode *node = malloc(sizeof(ASTNode));
    if (!node) return NULL;

    node->type = type;
    node->value = value ? strdup(value) : NULL;
    node->left = NULL;
    node->right = NULL;
    return node;
}

// === Public Function: Recursively free AST nodes ===
void free_ast(ASTNode *node) 
{
    if (!node) return;
    free_ast(node->left);
    free_ast(node->right);
    free(node->value);
    free(node);
}

// === Private Helper: Convert node type to string for printing ===
static const char *ast_type_to_string(ASTNodeType type) 
{
    switch (type) {
        case AST_FUNCTION_DEF:    return "FunctionDef";
        case AST_CALL_EXPR:       return "CallExpr";
        case AST_STRING_LITERAL:  return "StringLiteral";
        case AST_RETURN_STMT:     return "ReturnStmt";
        case AST_LITERAL:         return "Literal";
        case AST_DECLARATION:     return "Declaration";
        case AST_ASSIGNMENT:      return "Assignment";
        case AST_IDENTIFIER:      return "Identifier";
        case AST_INTEGER_LITERAL: return "IntegerLiteral";
        case AST_STATEMENT_LIST:  return "StatementList";
        default:                  return "Unknown";
    }
}

// === Private Helper: Recursively print AST structure ===
static void print_ast_recursive(ASTNode *node, int indent) 
{
    if (!node) return;

    for (int i = 0; i < indent; i++) printf("  ");

    printf("%s", ast_type_to_string(node->type));
    if (node->value) printf(": %s", node->value);
    printf("\n");

    print_ast_recursive(node->left, indent + 1);
    print_ast_recursive(node->right, indent + 1);
}

// === Public Function: Print an AST ===
void print_ast(ASTNode *root) {
    print_ast_recursive(root, 0);
}

