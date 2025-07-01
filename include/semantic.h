#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "ast.h"

// Enumerate the types of semantic results
typedef enum {
    SEMANTIC_OK = 0,
    SEMANTIC_ERROR
} SemanticResult;

// Analyze the AST for semantic errors
SemanticResult analyze(ASTNode *root);

#endif
