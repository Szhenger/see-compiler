#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "ast.h"

typedef enum {
    SEMANTIC_OK = 0,
    SEMANTIC_ERROR
} SemanticResult;

SemanticResult analyze(ASTNode *root);

#endif
