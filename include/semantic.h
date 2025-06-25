#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "ast.h"

// === Semantic Analysis Result Codes ===
// SEMANTIC_OK    → No semantic issues found
// SEMANTIC_ERROR → One or more semantic errors encountered
typedef enum {
    SEMANTIC_OK = 0,
    SEMANTIC_ERROR
} SemanticResult;

// === Public Function: Perform semantic analysis ===
// Traverses the AST and validates declarations, assignments, returns, etc.
// Returns SEMANTIC_OK if all checks pass, otherwise SEMANTIC_ERROR.
SemanticResult analyze(ASTNode *root);

#endif // SEMANTIC_H
