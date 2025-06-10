#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "semantic.h"

// Minimal test runner
void assert_semantic_ok(int result, const char *label) {
    if (!result) {
        fprintf(stderr, "FAIL: Semantic analysis failed for %s\n", label);
        exit(1);
    } else {
        printf("PASS: Semantic analysis passed for %s\n", label);
    }
}

int main(void) {
    // Simulate AST for: int main(void) { printf("Hello, world!\n"); return 0; }

    ASTNode *string_literal = create_ast_node(AST_STRING_LITERAL, "Hello, world!\\n");
    ASTNode *call_expr = create_ast_node(AST_CALL_EXPR, "printf");
    call_expr->left = string_literal;

    ASTNode *ret_value = create_ast_node(AST_LITERAL, "0");
    ASTNode *return_stmt = create_ast_node(AST_RETURN_STMT, NULL);
    return_stmt->left = ret_value;

    ASTNode *func = create_ast_node(AST_FUNCTION_DEF, "main");
    func->left = call_expr;
    func->right = return_stmt;

    // Run semantic analysis
    int result = analyze_semantics(func);
    assert_semantic_ok(result, "hello world");

    // Cleanup
    free_ast(func);

    return 0;
}
