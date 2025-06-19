#include <stdio.h>
#include <stdlib.h>
#include "ast.h"
#include "codegen.h"

// Code Generation Module
// ----------------------
// Translates the AST into x86-64 assembly (Intel syntax).
// Target output assumes Linux system with System V AMD64 calling convention.
// Currently supports a minimal subset: a single function that calls `printf`
// and returns an integer literal.

// === Emit function prologue ===
// Sets up stack frame: pushes base pointer and aligns stack
void generate_prologue(FILE *out) 
{
    fprintf(out,
        "    .intel_syntax noprefix\n"  // Use Intel assembly syntax
        "    .globl main\n"             // Declare global main label
        "main:\n"
        "    push rbp\n"                // Save base pointer
        "    mov rbp, rsp\n"            // Establish new stack frame
    );
}

// === Emit function epilogue ===
// Restores stack frame and returns from function
void generate_epilogue(FILE *out) 
{
    fprintf(out,
        "    mov rsp, rbp\n"            // Restore stack pointer
        "    pop rbp\n"                 // Restore base pointer
        "    mov rax, 0\n"              // Return value (default 0)
        "    ret\n"                     // Return from main
    );
}

// === Emit .rodata section for string literal ===
// Writes the string literal into the read-only data section
void generate_string_literal(FILE *out, const char *label, const char *str) 
{
    fprintf(out,
        "    .section .rodata\n"        // Switch to read-only data section
        "%s:\n"                         // Define label (e.g., .LC0)
        "    .string %s\n"              // Emit null-terminated string
        "    .text\n",                  // Switch back to code section
        label, str);
}

// === Emit code for call expression (e.g., printf) ===
// Assumes a single string literal argument and hardcodes label usage
void generate_call(FILE *out, ASTNode *node) 
{
    if (node->type != AST_CALL_EXPR) return;

    // Get argument (must be string literal)
    ASTNode *arg = node->left;
    if (!arg || arg->type != AST_STRING_LITERAL) return;

    const char *label = ".LC0"; // Hardcoded label for now

    generate_string_literal(out, label, arg->value);

    // Pass address of string literal in rdi (1st argument register)
    fprintf(out,
        "    lea rdi, %s\n"             // Load effective address of label
        "    call printf\n",            // Call external printf function
        label);
}

// === Emit code for return statement (return <int>) ===
// Moves the return value into eax (System V ABI: return register)
void generate_return(FILE *out, ASTNode *node) 
{
    if (node->type != AST_RETURN_STMT) return;

    ASTNode *val = node->left;
    if (!val) return;

    int return_val = atoi(val->value);  // Convert string to integer
    fprintf(out,
        "    mov eax, %d\n", return_val); // Set return value
}

// === Recursively walk AST and emit assembly code ===
// Traverses AST using a simple top-down strategy
void generate_code(FILE *out, ASTNode *node) 
{
    if (!node) return;

    switch (node->type) {
        case AST_FUNCTION_DEF:
            generate_prologue(out);
            generate_code(out, node->left);   // e.g., call expression
            generate_code(out, node->right);  // e.g., return statement
            generate_epilogue(out);
            break;

        case AST_CALL_EXPR:
            generate_call(out, node);
            break;

        case AST_RETURN_STMT:
            generate_return(out, node);
            break;

        default:
            // Other node types are ignored for now
            break;
    }
}

