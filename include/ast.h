#ifndef AST_H
#define AST_H

// Enumerates the type of AST nodes supoorted i.e. C/C++ semantics
typedef enum {
    // Primatives
    AST_FUNCTION_DEF,
    AST_STATEMENT_LIST,
    AST_DECLARATION,
    AST_ASSIGNMENT,
    AST_RETURN_STMT,
    AST_CALL_EXPR,

    // Names and Values
    AST_IDENTIFIER,
    AST_INTEGER_LITERAL,
    AST_STRING_LITERAL,
    AST_CHAR_LITERAL,
    AST_BINARY_OP,

    // Branching and Iteration
    AST_IF_STMT,
    AST_WHILE_LOOP,
    AST_FOR_LOOP,

    // Arrays
    AST_ARRAY_ACCESS,
    AST_ARRAY_DECLARATION
} ASTNodeType;

// Defines an AST node 
typedef struct ASTNode {
    ASTNodeType type;      // Node Type Field
    char *value;           // Node Value Field 
    struct ASTNode *left;  // Node Left Dependencies 
    struct ASTNode *right; // Node Right Dependencies 
} ASTNode;

// == Public API for AST Construction ==
ASTNode *create_ast_node(ASTNodeType type, const char *value);
void free_ast(ASTNode *node); 
void print_ast(ASTNode *root); // Debugging Support

#endif


