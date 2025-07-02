#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "codegen.h"
#include "ir.h"

// == Internal Variable Structure ==
typedef struct VarEntry {
    char *name;
    int offset;         
    struct VarEntry *next;
} VarEntry;

// == Internal Variable Table ==
static VarEntry *var_table = NULL;
static int current_offset = 0; 

// == Private Helper: Adds variable to table ==
static void add_variable(const char *name) 
{
    VarEntry *entry = malloc(sizeof(VarEntry));
    entry->name = strdup(name);
    current_offset -= 8;
    entry->offset = current_offset;
    entry->next = var_table;
    var_table = entry;
}

// == Private Helper: Finds the offset of instruction == 
static int find_variable_offset(const char *name) 
{
    for (VarEntry *e = var_table; e != NULL; e = e->next) {
        if (strcmp(e->name, name) == 0) return e->offset;
    }
    return 0;
}

// == Private Helper: Frees the variable table ==
static void free_var_table(void) 
{
    VarEntry *curr = var_table;
    while (curr) {
        VarEntry *next = curr->next;
        free(curr->name);
        free(curr);
        curr = next;
    }
    var_table = NULL;
    current_offset = 0;
}

// == Private Helper: Generates the required start of assembly file == 
static void generate_prologue(FILE *out) 
{
    fprintf(out,
        "    .intel_syntax noprefix\n"
        "    .globl main\n"
        "main:\n"
        "    push rbp\n"
        "    mov rbp, rsp\n"
    );

    if (current_offset != 0) {
        fprintf(out,
            "    sub rsp, %d\n", -current_offset);
    }
}
// == Private Helper: Generattes the required end of aasembly file
static void generate_epilogue(FILE *out) 
{
    fprintf(out,
        "    mov rsp, rbp\n"
        "    pop rbp\n"
        "    ret\n"
    );
}

// == Private Helper: Escape the string i.e. remove char/string intitialization syntax ==
static char *escape_string(const char *str) 
{
    size_t len = strlen(str);
    char *escaped = malloc(len * 2 + 1);
    if (!escaped) return NULL;

    char *dst = escaped;
    for (size_t i = 0; i < len; i++) {
        if (str[i] == '"' || str[i] == '\\') {
            *dst++ = '\\';
            *dst++ = str[i];
        } else if (str[i] == '\n') {
            *dst++ = '\\';
            *dst++ = 'n';
        } else {
            *dst++ = str[i];
        }
    }
    *dst = '\0';
    return escaped;
}

// == Private Helper: Creates the instruction label ==
static int label_counter = 0;
static const char *generate_string_label(const char *str, FILE *out) 
{
    static char label[32];
    snprintf(label, sizeof(label), ".LC%d", label_counter++);

    char *escaped = escape_string(str);
    if (!escaped) return NULL;

    fprintf(out,
        "    .section .rodata\n"
        "%s:\n"
        "    .string \"%s\"\n"
        "    .text\n",
        label, escaped);

    free(escaped);
    return strdup(label);
}

// == Public Function: Generates the assembly code in file out from ir list ==
void generate_code(FILE *out, IRInstr *ir) 
{
    if (!ir) return;

    for (IRInstr *curr = ir; curr != NULL; curr = curr->next) {
        if (curr->type == IR_DECL && curr->arg) {
            add_variable(curr->arg);
        }
    }

    generate_prologue(out);

    for (IRInstr *curr = ir; curr != NULL; curr = curr->next) {
        switch (curr->type) {
            case IR_PUSH:
                if (curr->arg[0] == '"' || strchr(curr->arg, ' ')) {
                    const char *label = generate_string_label(curr->arg, out);
                    fprintf(out, "    lea rax, %s\n", label);
                    fprintf(out, "    push rax\n");
                    free((char *)label);
                } else {
                    fprintf(out, "    mov rax, %s\n", curr->arg);
                    fprintf(out, "    push rax\n");
                }
                break;
            
            case IR_CALL:
                fprintf(out, "    call %s\n", curr->arg);
                break;

            case IR_DECL:
                break;

            case IR_LOAD: {
                int offset = find_variable_offset(curr->arg);
                if (offset != 0) {
                    fprintf(out, "    mov rax, QWORD PTR [rbp%+d]\n", offset);
                } else {
                    fprintf(out, "    mov eax, %s\n", curr->arg);
                }
                break;
            }

            case IR_STORE: {
                int offset = find_variable_offset(curr->arg);
                if (offset != 0) {
                    fprintf(out, "    mov QWORD PTR [rbp%+d], rax\n", offset);
                } else {
                    fprintf(stderr, "Warning: unknown variable '%s' in store\n", curr->arg);
                }
                break;
            }

            case IR_RET:
                generate_epilogue(out);
                break;

            case IR_ADD:
                fprintf(out, "    pop rbx\n");
                fprintf(out, "    pop rax\n");
                fprintf(out, "    add rax, rbx\n");
                fprintf(out, "    push rax\n");
                break;
            
            case IR_SUB:
                fprintf(out, "    pop rbx\n");
                fprintf(out, "    pop rax\n");
                fprintf(out, "    sub rax, rbx\n");
                fprintf(out, "    push rax\n");
                break;
            
            case IR_MUL:
                fprintf(out, "    pop rbx\n");
                fprintf(out, "    pop rax\n");
                fprintf(out, "    imul rax, rbx\n");
                fprintf(out, "    push rax\n");
                break;
            
            case IR_DIV:
                fprintf(out, "    pop rbx\n");         
                fprintf(out, "    pop rax\n");        
                fprintf(out, "    cqo\n");             
                fprintf(out, "    idiv rbx\n");        
                fprintf(out, "    push rax\n");
                break;
            
            case IR_LABEL:
                if (curr->arg && strcmp(curr->arg, "main") != 0 && strcmp(curr->arg, "entry") != 0) {
                    fprintf(out, "%s:\n", curr->arg);
                }
                break;
            
            case IR_JUMP:
                fprintf(out, "    jmp %s\n", curr->arg);
                break;
            
            case IR_JUMP_IF_ZERO:
                fprintf(out, "    pop rax\n");
                fprintf(out, "    cmp rax, 0\n");
                fprintf(out, "    je %s\n", curr->arg);
                break;
            
            case IR_CMP:
                fprintf(out, "    pop rbx\n");
                fprintf(out, "    pop rax\n");
                fprintf(out, "    cmp rax, rbx\n");
                break;

            default:
                break;
        }
    }

    free_var_table();
}


