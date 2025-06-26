#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "codegen.h"
#include "ir.h"

// === Symbol table for local variables (simple) ===
typedef struct VarEntry {
    char *name;
    int offset;          // Offset from rbp, e.g., -4, -8, ...
    struct VarEntry *next;
} VarEntry;

static VarEntry *var_table = NULL;
static int current_offset = 0;  // Negative offsets grow downward from rbp

// === Private Helper: Add a variable to the table with stack slot ===
static void add_variable(const char *name) {
    VarEntry *entry = malloc(sizeof(VarEntry));
    entry->name = strdup(name);
    current_offset -= 8; // allocate 8 bytes per var (64-bit)
    entry->offset = current_offset;
    entry->next = var_table;
    var_table = entry;
}

// === Private Helper: Find variable offset by name ===
// Returns offset or 0 if not found
static int find_variable_offset(const char *name) {
    for (VarEntry *e = var_table; e != NULL; e = e->next) {
        if (strcmp(e->name, name) == 0) return e->offset;
    }
    return 0; // not found
}

// === Private Helper: Free variable table ===
static void free_var_table(void) {
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

// === Private Helper: Emit function prologue with stack allocation ===
static void generate_prologue(FILE *out) {
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

// === Private Helper: Emit function epilogue ===
static void generate_epilogue(FILE *out) {
    fprintf(out,
        "    mov rsp, rbp\n"
        "    pop rbp\n"
        "    ret\n"
    );
}

// === Private Helper: Escape string literals for assembly ===
static char *escape_string(const char *str) {
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

// === Private Helper: Generate unique label for string literals ===
static int label_counter = 0;
static const char *generate_string_label(const char *str, FILE *out) {
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

// === Public Function: Generate code for the IR sequence ===
void generate_code(FILE *out, IRInstr *ir) {
    if (!ir) return;

    // First pass: build var table from declarations
    for (IRInstr *curr = ir; curr != NULL; curr = curr->next) {
        if (curr->type == IR_DECL && curr->arg) {
            add_variable(curr->arg);
        }
    }

    generate_prologue(out);

    for (IRInstr *curr = ir; curr != NULL; curr = curr->next) {
        switch (curr->type) {
            case IR_LABEL:
                // Skip "main" and "entry" labels in codegen output
                if (strcmp(curr->arg, "main") != 0 && strcmp(curr->arg, "entry") != 0) {
                    fprintf(out, "%s:\n", curr->arg);
                }
                break;

            case IR_PUSH:
                // Handle string literals or integer literals
                if (curr->arg[0] == '"' || strchr(curr->arg, ' ')) {
                    const char *label = generate_string_label(curr->arg, out);
                    fprintf(out, "    lea rdi, %s\n", label);
                    free((char *)label);
                } else {
                    fprintf(out, "    mov eax, %s\n", curr->arg);
                }
                break;

            case IR_CALL:
                fprintf(out, "    call %s\n", curr->arg);
                break;

            case IR_DECL:
                // Already handled by var table & stack allocation
                break;

            case IR_LOAD: {
                int offset = find_variable_offset(curr->arg);
                if (offset != 0) {
                    fprintf(out, "    mov eax, DWORD PTR [rbp%+d]\n", offset);
                } else {
                    // fallback: treat as literal
                    fprintf(out, "    mov eax, %s\n", curr->arg);
                }
                break;
            }

            case IR_STORE: {
                int offset = find_variable_offset(curr->arg);
                if (offset != 0) {
                    fprintf(out, "    mov DWORD PTR [rbp%+d], eax\n", offset);
                } else {
                    // unknown var, ignore or error
                    fprintf(stderr, "Warning: unknown variable '%s' in store\n", curr->arg);
                }
                break;
            }

            case IR_RET:
                generate_epilogue(out);
                break;

            default:
                // No-op
                break;
        }
    }

    free_var_table();
}


