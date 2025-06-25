#ifndef AST_H
#define AST_H

// === AST Node Types ===
typedef enum {
    AST_FUNCTION_DEF,     // Function definition (e.g., int main() { ... })
    AST_CALL_EXPR,        // Function call (e.g., printf("hi"))
    AST_STRING_LITERAL,   // String literal
    AST_INTEGER_LITERAL,  // Integer literal
    AST_IDENTIFIER,       // Variable or function name
    AST_DECLARATION,      // Variable declaration (e.g., int x;)
    AST_ASSIGNMENT,       // Variable assignment (e.g., x = 42;)
    AST_RETURN_STMT,      // Return statement
    AST_BINARY_OP,        // Binary operation (optional, for later)
    AST_STATEMENT_LIST,   // A list of statements (linked via `left` and `right`)
    AST_LITERAL           // General literal (e.g., "42" or variable name)
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


