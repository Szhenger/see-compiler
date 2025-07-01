#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "ir.h"

ASTNode *create_sample_ast(void) 
{
    ASTNode *str_lit = create_ast_node(AST_STRING_LITERAL, "hello, world!\\n");
    ASTNode *call = create_ast_node(AST_CALL_EXPR, "printf");
    call->left = str_lit;

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
    ASTNode *ast = create_sample_ast();
    IRInstr *ir = generate_ir(ast);
    assert(ir != NULL);

    IRInstr *curr = ir;
    int step = 0;

    while (curr) {
        switch (step) {
            case 0:
                assert(curr->type == IR_LABEL);
                assert(strcmp(curr->arg, "entry") == 0);
                break;
            case 1:
                assert(curr->type == IR_PUSH);
                assert(strcmp(curr->arg, "hello, world!\\n") == 0);
                break;
            case 2:
                assert(curr->type == IR_CALL);
                assert(strcmp(curr->arg, "printf") == 0);
                break;
            case 3:
                assert(curr->type == IR_PUSH);
                assert(strcmp(curr->arg, "0") == 0);
                break;
            case 4:
                assert(curr->type == IR_RET);
                assert(curr->arg == NULL);
                break;
            default:
                fprintf(stderr, "Unexpected extra IR instruction\n");
                exit(1);
        }
        curr = curr->next;
        step++;
    }

    assert(step == 5);
    printf("IR generation test passed.\n");

    free_ir(ir);
    free_ast(ast);
    return 0;
}

