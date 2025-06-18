#include <stdio.h>
#include <stdlib.h>
#include "ast.h"
#include "ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic.h"
#include "token.h"

// Simple Source Provider for Now
const char *load_sample_source(void) 
{
    return "int main(void) { printf(\"hello, world!\\n\"); return 0; }";
}

// Drives the Compilation of Source File
int main(void) 
{
    // Procedure 0: Get Source File
    const char *source = load_sample_source();

    // Procedure 1: Lexical Analysis
    printf("== Lexical Analysis ==\n");
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens || token_count == 0) {
        fprintf(stderr, "Lexing failed!\n");
        return 1;
    }

    // Procedure 2: Parsing
    printf("== Parsing ==\n");
    Parser *parser = init_parser(tokens, token_count);
    ASTNode *ast = parse(parser);
    if (!ast) {
        fprintf(stderr, "Parsing failed!\n");
        free_parser(parser);
        free_tokens(tokens, token_count);
        return 2;
    }

    // Procedure 3: Semantic Analysis
    printf("== Semantic Analysis ==\n");
    if (analyze(ast) != SEMANTIC_OK) {
        fprintf(stderr, "Semantic analysis failed!\n");
        free_ast(ast);
        free_parser(parser);
        free_tokens(tokens, token_count);
        return 3;
    }

    // Procedure 4: Debug AST Output
    printf("== Abstract Syntax Tree ==\n");
    print_ast(ast);
    
    // Procedure 5: IR Generation
    printf("== IR Generation ==\n");
    IRInstr *ir = generate_ir(ast);
    if (!ir) {
        fprintf(stderr, "IR generation failed!\n");
        free_ast(ast);
        free_parser(parser);
        free_tokens(tokens, token_count);
        return 4;
    }

    // Procedure 6: Debug IR Output
    printf("== Intermediate Representation ==\n");
    print_ir(ir);

    // Procedure 7: Code Generation
    printf("== x86 Code Generation ==\n");
    FILE *out = fopen("output.s", "w");
    if (!out) {
        fprintf(stderr, "Failed to open output file\n");
        free_ir(ir);
        free_ast(ast);
        free_parser(parser);
        free_tokens(tokens, token_count);
        return 5;    
    }
    generate_code(out, ir);
    fclose(out);

    // Procedure 8: Cleanup
    free_ir(ir);
    free_ast(ast);
    free_parser(parser);
    free_tokens(tokens, token_count);

    return 0;
}

