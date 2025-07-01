#ifndef CODEGEN_H
#define CODEGEN_H

#include <stdio.h>
#include "ir.h"

// Generates the x86 code instructions to an output file from a lidt of ir instructions
void generate_code(FILE *out, IRInstr *ir);

#endif
