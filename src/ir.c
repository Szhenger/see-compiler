#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "ir.h"

// == Public Function: Creates an ir instruction ==
IRInstr *create_ir_instr(IRType type, const char *arg) 
{
    IRInstr *instr = malloc(sizeof(IRInstr));
    if (!instr) return NULL;

    instr->type = type;
    instr->arg = arg ? strdup(arg) : NULL;
    instr->next = NULL;
    return instr;
}

// == Public Function: Destroys the space storing the ir instruction ==
void free_ir(IRInstr *head) 
{
    while (head) {
        IRInstr *next = head->next;
        free(head->arg);
        free(head);
        head = next;
    }
}

// == Public Function: Returns string representation of an input ir type == 
const char *ir_type_to_string(IRType type) 
{
    switch (type) {
        case IR_LABEL:         return "LABEL";
        case IR_CALL:          return "CALL";
        case IR_PUSH:          return "PUSH";
        case IR_RET:           return "RET";
        case IR_DECL:          return "DECL";
        case IR_STORE:         return "STORE";
        case IR_LOAD:          return "LOAD";
        case IR_ADD:           return "ADD";
        case IR_SUB:           return "SUB";
        case IR_MUL:           return "MUL";
        case IR_DIV:           return "DIV";
        case IR_JUMP:          return "JUMP";
        case IR_CMP:           return "CMP";
        case IR_JUMP_IF_ZERO:  return "JUMP_IF_ZERO"; 
        default:               return "UNKNOWN";
    }
}

// == Public Function: Prints the input ir linked list to the terminal for debugging ==
void print_ir(IRInstr *head) 
{
    for (IRInstr *curr = head; curr != NULL; curr = curr->next) {
        printf("%s", ir_type_to_string(curr->type));
        if (curr->arg) printf(" %s", curr->arg);
        printf("\n");
    }
}

// == Private Helper: Sets up the ir instruction from an AST node ==
void emit_ir_node(ASTNode *node, IRInstr **tail) 
{
    if (!node) return;

    switch (node->type) {
        case AST_FUNCTION_DEF:
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, node->value);
            emit_ir_node(node->left, tail);
            break;

        case AST_STATEMENT_LIST:
            emit_ir_node(node->left, tail);
            emit_ir_node(node->right, tail);
            break;

        case AST_DECLARATION:
            *tail = (*tail)->next = create_ir_instr(IR_DECL, node->value);
            break;

        case AST_ASSIGNMENT:
            emit_ir_node(node->right, tail);
            *tail = (*tail)->next = create_ir_instr(IR_STORE, node->left->value);
            break;

        case AST_INTEGER_LITERAL:
        case AST_STRING_LITERAL:
            *tail = (*tail)->next = create_ir_instr(IR_PUSH, node->value);
            break;
        
        case AST_IDENTIFIER:
            *tail = (*tail)->next = create_ir_instr(IR_LOAD, node->value);
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

        case AST_BINARY_OP:
            emit_ir_node(node->left, tail);
            emit_ir_node(node->right, tail);
            if (strcmp(node->value, "+") == 0)
                *tail = (*tail)->next = create_ir_instr(IR_ADD, NULL);
            else if (strcmp(node->value, "-") == 0)
                *tail = (*tail)->next = create_ir_instr(IR_SUB, NULL);
            else if (strcmp(node->value, "*") == 0)
                *tail = (*tail)->next = create_ir_instr(IR_MUL, NULL);
            else if (strcmp(node->value, "/") == 0)
                *tail = (*tail)->next = create_ir_instr(IR_DIV, NULL);
            else if (strcmp(node->value, "==") == 0 || strcmp(node->value, "!=") == 0 ||
                     strcmp(node->value, "<") == 0 || strcmp(node->value, ">") == 0 ||
                     strcmp(node->value, "<=") == 0 || strcmp(node->value, ">=") == 0)
                *tail = (*tail)->next = create_ir_instr(IR_CMP, node->value);
            break;

        case AST_IF_STMT: {
            static int label_id = 0;
            char else_label[32], end_label[32];
            sprintf(else_label, "else_%d", label_id);
            sprintf(end_label, "endif_%d", label_id);
            label_id++;
        
            emit_ir_node(node->left, tail);
            *tail = (*tail)->next = create_ir_instr(IR_JUMP_IF_ZERO, else_label);
            emit_ir_node(node->right->left, tail);
            *tail = (*tail)->next = create_ir_instr(IR_JUMP, end_label);
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, else_label);
            if (node->right->right) emit_ir_node(node->right->right, tail);
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, end_label);
            break;
        }

        case AST_WHILE_LOOP: {
            static int label_id = 0;
            char cond_label[32], end_label[32];
            sprintf(cond_label, "while_cond_%d", label_id);
            sprintf(end_label, "while_end_%d", label_id);
            label_id++;
        
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, cond_label);
            emit_ir_node(node->left, tail);
            *tail = (*tail)->next = create_ir_instr(IR_JUMP_IF_ZERO, end_label);
            emit_ir_node(node->right, tail);
            *tail = (*tail)->next = create_ir_instr(IR_JUMP, cond_label);
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, end_label);
            break;
        }

        case AST_FOR_LOOP: {
            static int label_id = 0;
            char cond_label[32], loop_label[32], end_label[32];
            sprintf(cond_label, "for_cond_%d", label_id);
            sprintf(loop_label, "for_loop_%d", label_id);
            sprintf(end_label, "for_end_%d", label_id);
            label_id++;
        
            if (node->left) {
                emit_ir_node(node->left, tail);
            }
            *tail = (*tail)->next = create_ir_instr(IR_JUMP, cond_label);
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, loop_label);
            if (node->right && node->right->right && node->right->right->right) {
                emit_ir_node(node->right->right->right, tail);
            }
            if (node->right && node->right->right && node->right->right->left) {
                emit_ir_node(node->right->right->left, tail);
            }
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, cond_label);
            if (node->right && node->right->left) {
                emit_ir_node(node->right->left, tail);
            }
            *tail = (*tail)->next = create_ir_instr(IR_JUMP_IF_ZERO, end_label);
            *tail = (*tail)->next = create_ir_instr(IR_JUMP, loop_label);
            *tail = (*tail)->next = create_ir_instr(IR_LABEL, end_label);
    
            break;
        }

        default:
            break;
    }
}

// == Public Function: Creates the ir linked list of instructions from an AST ==
IRInstr *generate_ir(ASTNode *ast) 
{
    IRInstr *head = create_ir_instr(IR_LABEL, "entry");
    IRInstr *tail = head;

    emit_ir_node(ast, &tail);
    return head;
}

