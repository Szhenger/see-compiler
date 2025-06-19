#ifndef AST_H
#define AST_H

// Abstract Syntax Tree (AST) Module
// ---------------------------------
// This module defines the structure and basic operations for the AST used to 
// represent the parsed structure of a program. The AST is constructed by the 
// parser and later analyzed semantically and translated into IR.

// Enumerates AST node types
typedef enum {
    AST_FUNCTION_DEF,     // Function definition (e.g., int main() { ... })
    AST_CALL_EXPR,        // Function call (e.g., printf("..."))
    AST_STRING_LITERAL,   // String literal (e.g., "hello")
    AST_RETURN_STMT,      // Return statement (e.g., return 0;)
    AST_LITERAL           // Integer literal (e.g., 0, 42)
} ASTNodeType;

// Represents a single AST node
typedef struct ASTNode {
    ASTNodeType type;         // The type of AST node
    char *value;              // The string value (e.g., literal, identifier)
    struct ASTNode *left;     // Left child (used for expressions, args, etc.)
    struct ASTNode *right;    // Right child (used for sequencing or nesting)
} ASTNode;

// Create a new AST node
// Allocates and initializes a node with the given type and value.
ASTNode *create_ast_node(ASTNodeType type, const char *value);

// Recursively free an AST subtree
void free_ast(ASTNode *node);

// Pretty-print the AST structure
void print_ast(ASTNode *root);

#endif

