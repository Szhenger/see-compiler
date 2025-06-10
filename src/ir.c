#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "ir.h"

// IR instruction structure
IRInstr *create_ir_instr(IRType type, const char *arg) {
    IRInstr *instr = malloc(sizeof(IRInstr));
    if (!instr) return NULL;
    instr->type = type;
    instr->arg = arg ? strdup(arg) : NULL;
    instr->next = NULL;
    return instr;
}

void free_ir(IRInstr *head) {
    while (head) {
        IRInstr *next = head->next;
        free(head->arg);
        free(head);
        head = next;
    }
}

const char *ir_type_to_string(IRType type) {
    switch (type) {
        case IR_LABEL: return "LABEL";
        case IR_CALL: return "CALL";
        case IR_PUSH: return "PUSH";
        case IR_RET: return "RET";
        default: return "UNKNOWN";
    }
}

void print_ir(IRInstr *head) {
    for (IRInstr *curr = head; curr != NULL; curr = curr->next) {
        printf("%s", ir_type_to_string(curr->type));
        if (curr->arg) printf(" %s", curr->arg);
        printf("\n");
    }
}

// Recursively generate IR from AST
void emit_ir_node(ASTNode *node, IRInstr **tail) {
    if (!node) return;

    switch (node->type) {
        case AST_FUNCTION_DEF:
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, node->value);
            emit_ir_node(node->left, tail);
            emit_ir_node(node->right, tail);
            break;

        case AST_CALL_EXPR:
            if (node->left && node->left->type == AST_STRING_LITERAL) {
                *tail = (*tail)->next = create_ir_instr(IR_PUSH, node->left->value);
            }
            *tail = (*tail)->next = create_ir_instr(IR_CALL, node->value);
            break;

        case AST_RETURN_STMT:
            if (node->left) {
                *tail = (*tail)->next = create_ir_instr(IR_PUSH, node->left->value);
            }
            *tail = (*tail)->next = create_ir_instr(IR_RET, NULL);
            break;

        default:
            break;
    }
}

IRInstr *generate_ir(ASTNode *ast) {
    IRInstr *head = create_ir_instr(IR_LABEL, "entry");  // optional entry label
    IRInstr *tail = head;
    emit_ir_node(ast, &tail);
    return head;
}
