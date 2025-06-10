#include <stdio.h>
#include <stdlib.h>
#include "ast.h"

void generate_prologue(FILE *out) 
{
    fprintf(out,
        "    .intel_syntax noprefix\n"
        "    .globl main\n"
        "main:\n"
        "    push rbp\n"
        "    mov rbp, rsp\n"
    );
}

void generate_epilogue(FILE *out) 
{
    fprintf(out,
        "    mov rsp, rbp\n"
        "    pop rbp\n"
        "    mov rax, 0\n"
        "    ret\n"
    );
}

// Generate code for string literals by putting them in .rodata
void generate_string_literal(FILE *out, const char *label, const char *str) 
{
    fprintf(out,
        "    .section .rodata\n"
        "%s:\n"
        "    .string %s\n"
        "    .text\n",
        label, str);
}

// Generate code for a call expression node (only printf for now)
void generate_call(FILE *out, ASTNode *node) 
{
    if (node->type != AST_CALL_EXPR) return;

    // Expect left child to be string literal
    ASTNode *arg = node->left;
    if (!arg || arg->type != AST_STRING_LITERAL) return;

    const char *label = ".LC0";

    generate_string_literal(out, label, arg->value);

    // Move address of string literal to rdi (first argument)
    fprintf(out,
        "    lea rdi, %s\n"
        "    call printf\n",
        label);
}

// Generate code for return statement (return <literal>)
void generate_return(FILE *out, ASTNode *node) 
{
    if (node->type != AST_RETURN_STMT) return;

    ASTNode *val = node->left;
    if (!val) return;

    int return_val = atoi(val->value);
    fprintf(out,
        "    mov eax, %d\n",
        return_val);
}

// Recursively generate code from AST
void generate_code(FILE *out, ASTNode *node) 
{
    if (!node) return;

    switch (node->type) {
        case AST_FUNCTION_DEF:
            generate_prologue(out);
            generate_code(out, node->left);  // call expression
            generate_code(out, node->right); // return statement
            generate_epilogue(out);
            break;

        case AST_CALL_EXPR:
            generate_call(out, node);
            break;

        case AST_RETURN_STMT:
            generate_return(out, node);
            break;

        default:
            // no-op for others now
            break;
    }
}
