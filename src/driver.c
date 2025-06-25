#include <stdio.h>
#include <stdlib.h>
#include "ast.h"
#include "ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic.h"
#include "token.h"

char *read_file(const char *filename) 
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open source: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    rewind(file);

    char *buffer = malloc(length + 1);
    if (!buffer) {
        fclose(file);
        fprintf(stderr, "Buffer memory allocation failed.\n");
        return NULL;
    }

    fread(buffer, 1, length, file);
    buffer[length] = '\0';
    fclose(file);
    return buffer;
}

int main(int argc, char **argv) 
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <source-file.c>\n", argv[0]);
        return 1;
    }

    // Procedure 1: Load Source File
    char *source = read_file(argv[1]);
    if (!source) return 1;

    // Procedure 2: Lexical Analysis
    printf("== Lexical Analysis ==\n");
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens || token_count == 0) {
        fprintf(stderr, "Lexing failed!\n");
        free(source);
        return 2;
    }

    // Procedure 3: Parsing
    printf("== Parsing ==\n");
    Parser *parser = init_parser(tokens, token_count);
    ASTNode *ast = parse(parser);
    if (!ast) {
        fprintf(stderr, "Parsing failed!\n");
        free_parser(parser);
        free_tokens(tokens, token_count);
        free(source);
        return 3;
    }

    // Procedure 4: Semantic Analysis
    printf("== Semantic Analysis ==\n");
    if (analyze(ast) != SEMANTIC_OK) {
        fprintf(stderr, "Semantic analysis failed!\n");
        free_ast(ast);
        free_parser(parser);
        free_tokens(tokens, token_count);
        free(source);
        return 4;
    }

    // Procedure 4.5: Debug AST Output
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
        free(source);
        return 5;
    }

    // Procedure 5.5: Debug IR Output
    printf("== Intermediate Representation ==\n");
    print_ir(ir);

    // Procedure 6: Code Generation
    printf("== x86 Code Generation ==\n");
    FILE *out = fopen("output.s", "w");
    if (!out) {
        fprintf(stderr, "Failed to open output file\n");
        free_ir(ir);
        free_ast(ast);
        free_parser(parser);
        free_tokens(tokens, token_count);
        free(source);
        return 6;    
    }
    generate_code(out, ir);
    fclose(out);

    // Procedure 7.5: Cleanup
    free_ir(ir);
    free_ast(ast);
    free_parser(parser);
    free_tokens(tokens, token_count);
    free(source);

    return 0;
}


