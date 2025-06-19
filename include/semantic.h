#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "ast.h"

typedef enum {
    SEMANTIC_OK = 0,
    SEMANTIC_ERROR
} SemanticResult;

// Performs semantic validation on the abstract syntax tree (AST)
SemanticResult analyze(ASTNode *root);

#endif
