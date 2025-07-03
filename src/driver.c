#include <stdio.h>
#include <stdlib.h>
#include "ast.h"
#include "ir.h"
#include "lexer.h"
#include "parser.h"
#include "semantic.h"
#include "token.h"

// Forward Helper Function Prototype
char *read_file(const char *filename);

// Main Function: Driving Compilation of Source File 
int main(int argc, char **argv) 
{
    if (argc != 2) {
        fprintf(stderr, "Proper Usage: %s <source-file.c>\n", argv[0]);
        return -1;
    }

    // Procedure 1: Get C Source File
    printf("== SeeCompilation ==\n");
    char *source = read_file(argv[1]);
    if (!source) return 1;

    // Procedure 2: Tokenize the C Source String
    printf("== Tokenizing Source File ==\n");
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens || token_count == 0) {
        fprintf(stderr, "Lexing failed!\n");
        free(source);
        return 2;
    }
    print_tokens(tokens, token_count);
    
    // Procedure 3: Parse the Token Stream
    printf("== Parsing Token Stream ==\n");
    Parser *parser = init_parser(tokens, token_count);
    ASTNode *ast = parse(parser);
    if (!ast) {
        fprintf(stderr, "Parsing failed!\n");
        free_parser(parser);
        free_tokens(tokens, token_count);
        free(source);
        return 3;
    }
    print_ast(ast);

    // Procedure 4: Run Semantic Analysis on AST
    if (analyze(ast) != SEMANTIC_OK) {
        fprintf(stderr, "Semantic analysis failed!\n");
        free_ast(ast);
        free_parser(parser);
        free_tokens(tokens, token_count);
        free(source);
        return 4;
    }
    
    // Procedure 5: Generate IR Instructions from AST
    printf("== Generating IR Instructions ==\n");
    IRInstr *ir = generate_ir(ast);
    if (!ir) {
        fprintf(stderr, "IR generation failed!\n");
        free_ast(ast);
        free_parser(parser);
        free_tokens(tokens, token_count);
        free(source);
        return 5;
    }
    print_ir(ir);

    // Procedure 6: Generate x86 Assembly Instructions from IR  
    printf("== Generating x86 Assembly Instructions ==\n");
    FILE *output = fopen("output.s", "w");
    if (!output) {
        fprintf(stderr, "Failed to open output file\n");
        free_ir(ir);
        free_ast(ast);
        free_parser(parser);
        free_tokens(tokens, token_count);
        free(source);
        return 6;    
    }
    generate_code(output, ir);
    fclose(output);

    // Procedure 7: Cleanup
    free_ir(ir);
    free_ast(ast);
    free_parser(parser);
    free_tokens(tokens, token_count);
    free(source);

    return 0;
}

// Helper Function: Read C Source File into Memory
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


