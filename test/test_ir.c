#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "ir.h"

// Helper: Create AST for printf("hello, world!\n");
ASTNode *create_sample_ast(void) 
{
    ASTNode *str_lit = create_ast_node(AST_STRING_LITERAL, "\"hello, world!\\n\"");
    ASTNode *call = create_ast_node(AST_CALL_EXPR, "printf");
    call->left = str_lit;

    ASTNode *ret_val = create_ast_node(AST_LITERAL, "0");
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

    // Generate IR
    IRInstr *ir_head = generate_ir(ast);
    if (!ir_head) {
        fprintf(stderr, "IR generation failed!\n");
        free_ast(ast);
        return 1;
    }

    // Check IR contents (basic example check)
    IRInstr *current = ir_head;
    int step = 0;

    // Example expected sequence:
    // step 0: PRINT "hello, world!\n"
    // step 1: RETURN 0
    while (current != NULL) {
        switch (step) {
            case 0:
                if (strcmp(current->opcode, "PRINT") != 0 || strcmp(current->operand, "\"hello, world!\\n\"") != 0) {
                    fprintf(stderr, "FAIL: Expected PRINT instruction with \"hello, world!\\n\"\n");
                    free_ir(ir_head);
                    free_ast(ast);
                    return 1;
                }
                break;
            case 1:
                if (strcmp(current->opcode, "RETURN") != 0 || strcmp(current->operand, "0") != 0) {
                    fprintf(stderr, "FAIL: Expected RETURN instruction with operand 0\n");
                    free_ir(ir_head);
                    free_ast(ast);
                    return 1;
                }
                break;
            default:
                fprintf(stderr, "FAIL: Unexpected extra IR instruction: %s %s\n", current->opcode, current->operand);
                free_ir(ir_head);
                free_ast(ast);
                return 1;
        }
        current = current->next;
        step++;
    }

    if (step != 2) {
        fprintf(stderr, "FAIL: IR instruction count mismatch, expected 2, got %d\n", step);
        free_ir(ir_head);
        free_ast(ast);
        return 1;
    }

    printf("IR unit test passed.\n");

    free_ir(ir_head);
    free_ast(ast);

    return 0;
}
