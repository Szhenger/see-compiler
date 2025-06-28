#ifndef CODEGEN_H
#define CODEGEN_H

#include <stdio.h>
#include "ir.h"

void generate_code(FILE *out, IRInstr *ir);

#endif
