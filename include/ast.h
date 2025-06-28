#ifndef AST_H
#define AST_H

// === AST Node Types ===
typedef enum {
    AST_FUNCTION_DEF,
    AST_STATEMENT_LIST,
    AST_DECLARATION,
    AST_ASSIGNMENT,
    AST_RETURN_STMT,
    AST_CALL_EXPR,
    AST_IDENTIFIER,
    AST_INTEGER_LITERAL,
    AST_STRING_LITERAL,
    AST_BINARY_OP,
    AST_IF_STMT,
    AST_WHILE_LOOP,
    AST_FOR_LOOP,
    AST_ARRAY_ACCESS,
    AST_ARRAY_DECLARATION
} ASTNodeType;

// === AST Node Structure ===
typedef struct ASTNode {
    ASTNodeType type;
    char *value;              // Optional value (e.g., identifier name, literal)
    struct ASTNode *left;     // Left child or first part
    struct ASTNode *right;    // Right child or next part
} ASTNode;

// === Public API ===
ASTNode *create_ast_node(ASTNodeType type, const char *value);
void free_ast(ASTNode *node);
void print_ast(ASTNode *root);

#endif // AST_H


