#ifndef IR_H
#define IR_H

#include "ast.h"

typedef enum {
    IR_LABEL,
    IR_CALL,
    IR_PUSH,
    IR_RET
} IRType;

typedef struct IRInstr {
    IRType type;
    char *arg;
    struct IRInstr *next;
} IRInstr;

IRInstr *create_ir_instr(IRType type, const char *arg);
IRInstr *generate_ir(ASTNode *ast);
void print_ir(IRInstr *head);
void free_ir(IRInstr *head);

#endif
