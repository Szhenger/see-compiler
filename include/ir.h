#ifndef IR_H
#define IR_H

#include "ast.h"

// Intermediate Representation (IR) Module
// ---------------------------------------
// This module defines a simple linear Intermediate Representation (IR) for a 
// restricted C-like language. The IR models operations such as function calls, 
// stack manipulation, and control flow.
//
// It is generated from the AST and consumed by the code generator to emit x86.

// === Enumerates IR instruction types ===
typedef enum {
    IR_LABEL,   // A label (e.g., function entry point)
    IR_CALL,    // A function call (e.g., printf)
    IR_PUSH,    // Push a literal value onto the stack
    IR_RET      // Return from function
} IRType;

// === Represents a single IR instruction ===
typedef struct IRInstr {
    IRType type;             // Instruction kind (label, call, etc.)
    char *arg;               // Argument (e.g., label name, literal string/int)
    struct IRInstr *next;    // Next instruction in the linear IR sequence
} IRInstr;

// === Creates a new IR instruction ===
IRInstr *create_ir_instr(IRType type, const char *arg);

// === Translates an AST into a linear IR representation ===
IRInstr *generate_ir(ASTNode *ast);

// === Prints the IR sequence to stdout (for debugging) ===
void print_ir(IRInstr *head);

// === Frees all dynamically allocated IR instructions ===
void free_ir(IRInstr *head);

#endif

