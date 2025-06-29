#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "lexer.h"
#include "parser.h"
#include "token.h"

void assert_ast_node(ASTNode *node, ASTNodeType expected_type, const char *expected_value) {
    assert(node != NULL);
    assert(node->type == expected_type);

    if (expected_value == NULL) {
        assert(node->value == NULL);
    } else {
        assert(strcmp(node->value, expected_value) == 0);
    }
}

int main(void) 
{
    const char *source = "int main(void) { printf(\"hello, world!\\n\"); return 0; }";

    int token_count = 0;
    Token *tokens = tokenize(source_code, &token_count);
    assert(tokens != NULL);

    Parser *parser = init_parser(tokens, &token_count);
    ASTNode *tree = parse(parser);
    assert(tree != NULL);

    assert_ast_node(tree, AST_FUNCTION_DEF, "main");

    ASTNode *call = tree->left;
    assert_ast_node(call, AST_CALL_EXPR, "printf");
    assert_ast_node(call->left, AST_STRING_LITERAL, "hello, world!\\n");

    ASTNode *ret = tree->right;
    assert_ast_node(ret, AST_RETURN_STMT, NULL);
    assert_ast_node(ret->left, AST_INTEGER_LITERAL, "0");

    printf("Parser test passed.\n");

    free_ast(tree);
    free_parser(parser);
    free_tokens(tokens, token_count);

    return 0;
}


