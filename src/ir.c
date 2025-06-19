#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "ir.h"

// Intermediate Representation (IR) Generation
// ------------------------------------------
// This module translates the Abstract Syntax Tree (AST) into a flat linked list
// of Intermediate Representation instructions. These IR instructions model key 
// control and data flow operations for backend consumption (e.g., codegen).

// === Create and initialize a new IR instruction ===
// - type: type of IR instruction (LABEL, CALL, PUSH, RET)
// - arg: optional argument string (e.g., label name, literal value)
IRInstr *create_ir_instr(IRType type, const char *arg) 
{
    IRInstr *instr = malloc(sizeof(IRInstr));
    if (!instr) return NULL;

    instr->type = type;
    instr->arg = arg ? strdup(arg) : NULL;
    instr->next = NULL;
    return instr;
}

// === Free a linked list of IR instructions ===
void free_ir(IRInstr *head) 
{
    while (head) {
        IRInstr *next = head->next;
        free(head->arg);
        free(head);
        head = next;
    }
}

// === Convert IR type enum to string (for printing/debugging) ===
const char *ir_type_to_string(IRType type) 
{
    switch (type) {
        case IR_LABEL: return "LABEL";
        case IR_CALL:  return "CALL";
        case IR_PUSH:  return "PUSH";
        case IR_RET:   return "RET";
        default:       return "UNKNOWN";
    }
}

// === Print a linked list of IR instructions ===
// Format: <TYPE> <ARG> (if any)
void print_ir(IRInstr *head) 
{
    for (IRInstr *curr = head; curr != NULL; curr = curr->next) {
        printf("%s", ir_type_to_string(curr->type));
        if (curr->arg) printf(" %s", curr->arg);
        printf("\n");
    }
}

// === Recursive IR emitter ===
// Traverses the AST and appends corresponding IR instructions to the tail.
void emit_ir_node(ASTNode *node, IRInstr **tail) 
{
    if (!node) return;

    switch (node->type) {
        case AST_FUNCTION_DEF:
            // Emit label for function name
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, node->value);

            // Emit IR for function body (sequential left/right walk)
            emit_ir_node(node->left, tail);   // first stmt
            emit_ir_node(node->right, tail);  // second stmt
            break;

        case AST_CALL_EXPR:
            // Push argument (if it's a string literal)
            if (node->left && node->left->type == AST_STRING_LITERAL) {
                *tail = (*tail)->next = create_ir_instr(IR_PUSH, node->left->value);
            }

            // Emit function call
            *tail = (*tail)->next = create_ir_instr(IR_CALL, node->value);
            break;

        case AST_RETURN_STMT:
            // Push return value (if any)
            if (node->left) {
                *tail = (*tail)->next = create_ir_instr(IR_PUSH, node->left->value);
            }

            // Emit return instruction
            *tail = (*tail)->next = create_ir_instr(IR_RET, NULL);
            break;

        default:
            // Unsupported AST node; ignore silently
            break;
    }
}

// === Generate IR from AST root ===
// Starts with an entry label and builds a linked list of IR nodes.
IRInstr *generate_ir(ASTNode *ast) 
{
    IRInstr *head = create_ir_instr(IR_LABEL, "entry");  // Optional initial label
    IRInstr *tail = head;

    emit_ir_node(ast, &tail);
    return head;
}

