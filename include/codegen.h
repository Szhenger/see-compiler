#ifndef CODEGEN_H
#define CODEGEN_H

#include <stdio.h>
#include "ir.h"

// Emits the full x86-64 assembly output from a sequence of IR instructions
void generate_code(FILE *out, IRInstr *ir);

#endif // CODEGEN_H
