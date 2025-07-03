#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"

// == Public Function: Creates AST node representation of input token == 
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

// == Public Function: Frees the AST ==
void free_ast(ASTNode *node) 
{
    if (!node) return;
    free_ast(node->left);
    free_ast(node->right);
    free(node->value);
    free(node);
}

// == Private Helper: Returns a string representation of AST node == 
static const char *ast_type_to_string(ASTNodeType type) 
{
    switch (type) {
        case AST_FUNCTION_DEF:    return "FunctionDef";
        case AST_CALL_EXPR:       return "CallExpr";
        case AST_STRING_LITERAL:  return "StringLiteral";
        case AST_RETURN_STMT:     return "ReturnStmt";
        case AST_EXPRESSION_STMT: return "ExpressStmt";
        case AST_DECLARATION:     return "Declaration";
        case AST_ASSIGNMENT:      return "Assignment";
        case AST_IDENTIFIER:      return "Identifier";
        case AST_INTEGER_LITERAL: return "IntegerLiteral";
        case AST_STATEMENT_LIST:  return "StatementList";
        case AST_BINARY_OP:       return "BinaryOp";
        case AST_CHAR_LITERAL:    return "CharLiteral";
        case AST_IF_STMT:         return "IfStmt";
        case AST_FOR_LOOP:        return "ForLoop";
        case AST_WHILE_LOOP:      return "WhileLoop";
        default:                  return "Unknown";
    }
}

// == Private Helper: Recursively prints AST ==
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

// Pulblic Function: Prints AST recursively ==
void print_ast(ASTNode *root) {
    printf("== Abstract Syntax Tree ==\n");
    print_ast_recursive(root, 0);
}

