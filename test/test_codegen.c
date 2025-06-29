#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "ir.h"

ASTNode *build_test_ast(void) 
{
    ASTNode *str = create_ast_node(AST_STRING_LITERAL, "hello, world!\\n");
    ASTNode *call = create_ast_node(AST_CALL_EXPR, "printf");
    call->left = str;

    ASTNode *ret_val = create_ast_node(AST_INTEGER_LITERAL, "0");
    ASTNode *ret_stmt = create_ast_node(AST_RETURN_STMT, NULL);
    ret_stmt->left = ret_val;

    ASTNode *func = create_ast_node(AST_FUNCTION_DEF, "main");
    func->left = call;
    func->right = ret_stmt;

    return func;
}

int main(void) 
{
    ASTNode *ast = build_test_ast();
    assert(ast != NULL);

    IRInstr *ir = generate_ir(ast);
    if (!ir) {
        fprintf(stderr, "IR generation failed\n");
        free_ast(ast);
        return 1;
    }

    char *buf = NULL;
    size_t size = 0;
    FILE *memstream = open_memstream(&buf, &size);
    if (!memstream) {
        fprintf(stderr, "Failed to open memory stream\n");
        free_ast(ast);
        free_ir(ir);
        return 1;
    }

    generate_code(memstream, ir);
    fflush(memstream);
    fclose(memstream);

    if (strstr(buf, "main:") &&
        strstr(buf, "call printf") &&
        strstr(buf, "push rax") &&
        strstr(buf, "ret")) {
        printf("Code generation test PASSED\n");
    } else {
        printf("Code generation test FAILED\n\n");
        printf("Generated assembly:\n%s\n", buf);
    }

    free(buf);
    free_ast(ast);
    free_ir(ir);
    return 0;
}

