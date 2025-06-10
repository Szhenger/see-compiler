#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"

// Forward declare your generate_code function
void generate_code(FILE *out, ASTNode *node);

// Helper: create a simple AST for `int main() { printf("hello, world!\n"); return 0; }`
ASTNode *build_test_ast(void) 
{
    // Build string literal node
    ASTNode *str_lit = create_ast_node(AST_STRING_LITERAL, "\"hello, world!\\n\"");

    // Build call expression node with the string literal as argument
    ASTNode *call = create_ast_node(AST_CALL_EXPR, "printf");
    call->left = str_lit;

    // Build return statement node returning 0
    ASTNode *ret_val = create_ast_node(AST_LITERAL, "0");
    ASTNode *ret_stmt = create_ast_node(AST_RETURN_STMT, NULL);
    ret_stmt->left = ret_val;

    // Build function definition node (main) with call and return as children
    ASTNode *func = create_ast_node(AST_FUNCTION_DEF, "main");
    func->left = call;
    func->right = ret_stmt;

    return func;
}

int main(void) 
{
    ASTNode *ast = build_test_ast();
    if (!ast) {
        fprintf(stderr, "Failed to build test AST\n");
        return 1;
    }

    // Open a memory stream to capture output (POSIX)
    FILE *memstream = open_memstream(NULL, NULL);
    if (!memstream) {
        fprintf(stderr, "Failed to open memory stream\n");
        free_ast(ast);
        return 1;
    }

    // Generate assembly code for the test AST
    generate_code(memstream, ast);

    // Flush the stream and get the generated assembly string
    fflush(memstream);
    long size;
    char *asm_code = NULL;
    asm_code = NULL;
    size = 0;
    fseek(memstream, 0, SEEK_END);
    size = ftell(memstream);
    fseek(memstream, 0, SEEK_SET);

    asm_code = malloc(size + 1);
    if (!asm_code) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(memstream);
        free_ast(ast);
        return 1;
    }
    fread(asm_code, 1, size, memstream);
    asm_code[size] = '\0';
    fclose(memstream);

    // Basic checks: verify key assembly instructions exist
    if (strstr(asm_code, "main:") &&
        strstr(asm_code, "call printf") &&
        strstr(asm_code, "mov eax, 0") &&
        strstr(asm_code, "ret")) {
        printf("Code generation test PASSED\n");
    } else {
        printf("Code generation test FAILED\n");
        printf("Generated assembly:\n%s\n", asm_code);
    }

    free(asm_code);
    free_ast(ast);

    return 0;
}
