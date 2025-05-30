#ifndef AST_H
#define AST_H

// Define the types of abstract syntax tree (AST) tokens
typedef enum {
    AST_FUNCTION_DEF,
    AST_RETURN_STMT,
    AST_STRING_LITERAL,
    AST_CALL_EXPR,
    AST_IDENTIFIER,
    AST_LITERAL,
    // Add more as needed
} ASTNodeType;

// Define the structure of abstract syntax tree (AST) nodes
typedef struct ASTNode {
    ASTNodeType type;
    struct ASTNode* left;
    struct ASTNode* right;
    char* value; // e.g. function name, string, etc.
} ASTNode;

// Utility function to create AST nodes
ASTNode* create_ast_node(ASTNodeType type, const char* value);

void free_ast(ASTNode* node);

#endif
