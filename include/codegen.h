#ifndef CODEGEN_H
#define CODEGEN_H

#include <stdio.h>
#include "ast.h"

// Generate the function prologue for x86-64 assembly output
void generate_prologue(FILE *out);

// Generate the function epilogue for x86-64 assembly output
void generate_epilogue(FILE *out);

// Generate assembly for string literals in .rodata section
void generate_string_literal(FILE *out, const char *label, const char *str);

// Generate assembly for a function call node (currently supports printf)
void generate_call(FILE *out, ASTNode *node);

// Generate assembly for a return statement node
void generate_return(FILE *out, ASTNode *node);

// Recursively generate assembly code for the entire AST
void generate_code(FILE *out, ASTNode *node);

#endif // CODEGEN_H
