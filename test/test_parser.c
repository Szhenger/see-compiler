#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "ast.h"
#include "lexer.h"
#include "parser.h"
#include "tokens.h"

// Assert helper for AST nodes
void assert_ast_node(ASTNode *node, ASTNodeType expected_type, const char *expected_value) 
{
    assert(node != NULL);
    assert(node->type == expected_type);
    assert(strcmp(node->value, expected_value) == 0);
}

// Main test driver
int main(void) 
{
    const char *source_code = "int main(void) { printf(\"hello, world!\\n\"); return 0; }";

    // Tokenize
    int token_count = 0;
    Token *tokens = tokenize(source_code, &token_count);
    assert(tokens != NULL);

    // Parse
    Parser *parser = init_parser(tokens, token_count);
    ASTNode *tree = parse(parser);
    assert(tree != NULL);

    // Assert function node
    assert_ast_node(tree, AST_FUNCTION_DEF, "main");

    // Assert call node: printf("Hello, world!\n")
    ASTNode *call = tree->left;
    assert_ast_node(call, AST_CALL_EXPR, "printf");
    assert_ast_node(call->left, AST_STRING_LITERAL, "Hello, world!\\n");

    // Assert return node: return 0
    ASTNode *ret = tree->right;
    assert_ast_node(ret, AST_RETURN_STMT, NULL);
    assert_ast_node(ret->left, AST_LITERAL, "0");

    printf("Parser test passed.\n");

    // Cleanup
    free_ast(tree);
    free_parser(parser);
    free_tokens(tokens, token_count);

    return 0;
}

