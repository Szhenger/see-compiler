#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "lexer.h"
#include "parser.h"
#include "semantic.h"
#include "token.h"

int main(void) {
    const char *source = "int main(void) { printf(\"hello, world!\\n\"); return 0; }";

    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    assert(tokens != NULL && token_count > 0);

    Parser *parser = init_parser(tokens, &token_count);
    assert(parser != NULL);
    ASTNode *ast = parse(parser);
    assert(ast != NULL);

    SemanticResult result = analyze(ast);
    assert(result == SEMANTIC_OK);

    printf("Semantic analysis test passed.\n");

    free_ast(ast);
    free_parser(parser);
    free_tokens(tokens, token_count);

    return 0;
}
