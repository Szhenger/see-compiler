#include <stdio.h>
#include <stdlib.h>
#include "ast.h"
#include "lexer.h"
#include "parser.h"
#include "tokens.h"

// Simple source provider for now
const char *load_sample_source(void) 
{
    return "int main(void) { printf(\"Hello, world!\\n\"); return 0; }";
}

int main(void) 
{
    const char *source = load_sample_source();

    // Procedure 1: Lexical Analysis
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens || token_count == 0) {
        fprintf(stderr, "Lexing failed!\n");
        return 1;
    }

    // Procedure 2: Parsing
    Parser *parser = init_parser(tokens, token_count);
    ASTNode *ast = parse(parser);
    if (!ast) {
        fprintf(stderr, "Parsing failed!\n");
        free_parser(parser);
        free_tokens(tokens, token_count);
        return 2;
    }

    // Procedure 3: Debug Output
    print_ast(ast);

    // Procedure 4: Cleanup
    free_ast(ast);
    free_parser(parser);
    free_tokens(tokens, token_count);

    return 0;
}

