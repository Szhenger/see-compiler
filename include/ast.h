#ifndef AST_H
#define AST_H

typedef enum {
    AST_FUNCTION_DEF,
    AST_CALL_EXPR,
    AST_STRING_LITERAL,
    AST_RETURN_STMT,
    AST_LITERAL
} ASTNodeType;

typedef struct ASTNode {
    ASTNodeType type;
    char *value;
    struct ASTNode *left;
    struct ASTNode *right;
} ASTNode;

ASTNode *create_ast_node(ASTNodeType type, const char *value);
void free_ast(ASTNode *node);
void print_ast(ASTNode *root);

#endif

