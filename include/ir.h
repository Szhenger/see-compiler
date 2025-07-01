#ifndef IR_H
#define IR_H

#include "ast.h"

// Enumerate the types of ir instructions supported
typedef enum {
    IR_LABEL,
    IR_CALL,
    IR_PUSH,
    IR_RET,
    IR_DECL,
    IR_STORE,
    IR_LOAD,
    IR_JUMP,
    IR_JUMP_IF_ZERO,
    IR_CMP,
    IR_ADD,
    IR_SUB,
    IR_MUL,
    IR_DIV
} IRType;

// Defines an ir instruction 
typedef struct IRInstr {
    IRType type;             
    char *arg;               
    struct IRInstr *next;    
} IRInstr;

// == Public API ==
IRInstr *create_ir_instr(IRType type, const char *arg);
IRInstr *generate_ir(ASTNode *ast);
void print_ir(IRInstr *head);
void free_ir(IRInstr *head);
const char *ir_type_to_string(IRType type); 

#endif

