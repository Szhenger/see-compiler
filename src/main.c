#include <stdio.h>
#include <stdlib.h>
#include "ast.h" 
#include "lexer.h"
#include "parser.h"
#include "token.h"

int main(void) 
{
    // Get the source file
    const char *source =
        "int main(void)" 
        "{\n"
        "    printf(\"Hello, world!\\n\");\n"
        "    return 0;\n"
        "}";

    // Procedure 1: Lexical Analysis
    TokenStream *tokens = tokenize(source_code);
    if (tokens == NULL) {
        fprintf(stderr, "Lexing failed!\n");
        return 1;
    }

    // Procedure 2: Syntactic Analysis (Parsing)
    ASTNode *root = parse(tokens);
    if (root == NULL) {
        fprintf(stderr, "Parsing failed!\n");
        return 1;
    }

    // Procedure 3 (optional): Debugging Output
    print_ast(root);

    // Procedure 4: Clean Up
    free_ast(root);
    free_token_stream(tokens);

    return 0;}
