#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lexer.h"
#include "token.h"

// Basic assertion helper
void assert_token(Token t, TokenType expected_type, const char *expected_lexeme) 
{
    if (t.type != expected_type || strcmp(t.lexeme, expected_lexeme) != 0) {
        fprintf(stderr, "FAIL: Got [%s, \"%s\"], expected [%s, \"%s\"]\n",
                token_type_to_string(t.type), t.lexeme,
                token_type_to_string(expected_type), expected_lexeme);
        exit(1);
    }
}

// Unit test for the lexer
int main(void) 
{
    const char *source = "int main(void) { return 0; }";
    int count = 0;

    Token *tokens = tokenize(source, &count);
    if (!tokens) {
        fprintf(stderr, "Lexer failed: NULL token array.\n");
        return 3;
    }

    // Token-by-token assertions
    int i = 0;
    assert_token(tokens[i++], TOKEN_KEYWORD, "int");
    assert_token(tokens[i++], TOKEN_IDENTIFIER, "main");
    assert_token(tokens[i++], TOKEN_SYMBOL, "(");
    assert_token(tokens[i++], TOKEN_KEYWORD, "void");
    assert_token(tokens[i++], TOKEN_SYMBOL, ")");
    assert_token(tokens[i++], TOKEN_SYMBOL, "{");
    assert_token(tokens[i++], TOKEN_KEYWORD, "return");
    assert_token(tokens[i++], TOKEN_INTEGER_LITERAL, "0");
    assert_token(tokens[i++], TOKEN_SYMBOL, ";");
    assert_token(tokens[i++], TOKEN_SYMBOL, "}");
    assert_token(tokens[i++], TOKEN_EOF, "");

    printf("Lexer test passed.\n");

    // Free all lexemes
    for (int j = 0; j < count; j++) {
        free(tokens[j].lexeme);
    }
    free(tokens);
    return 0;
}

