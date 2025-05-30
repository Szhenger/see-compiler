#include "lexer.h"
#include "parser.h"
#include "ast.h"
#include <stdio.h>

int main(void) 
{
    const char *source_code = "int main() { printf(\"Hello, world!\\n\"); return 0; }";
    Token* tokens;
    int token_count;
    tokens = tokenize(source_code, &token_count);

    Parser *parser = init_parser(tokens, token_count);
    ASTNode *tree = parse(parser);

    // For now, just print top-level structure
    printf("Function: %s\n", tree->value); // main
    printf("Call: %s with arg: %s\n", tree->left->value, tree->left->left->value); // printf
    printf("Return: %s\n", tree->right->left->value); // 0

    free_ast(tree);
    free_parser(parser);
    free_tokens(tokens, token_count);

    return 0;
}
