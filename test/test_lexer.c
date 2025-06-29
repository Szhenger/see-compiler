#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lexer.h"
#include "token.h"

void assert_token(Token t, TokenType expected_type, const char *expected_lexeme) 
{
    assert(t.type == expected_type);
    assert(strcmp(t.lexeme, expected_lexeme) == 0);
}

int main(void) 
{
    const char *source = "int main(void) { printf(\"hello, world!\\n\"); return 0; }";
    int count = 0;

    Token *tokens = tokenize(source, &count);
    assert(tokens != NULL);

    int i = 0;
    assert_token(tokens[i++], TOKEN_KEYWORD, "int");
    assert_token(tokens[i++], TOKEN_IDENTIFIER, "main");
    assert_token(tokens[i++], TOKEN_LPAREN, "(");
    assert_token(tokens[i++], TOKEN_KEYWORD, "void");
    assert_token(tokens[i++], TOKEN_RPAREN, ")");
    assert_token(tokens[i++], TOKEN_LBRACE, "{");
    assert_token(tokens[i++], TOKEN_IDENTIFIER, "printf");
    assert_token(tokens[i++], TOKEN_LPAREN, "(");
    assert_token(tokens[i++], TOKEN_STRING_LITERAL, "hello, world!\\n");
    assert_token(tokens[i++], TOKEN_RPAREN, ")");
    assert_token(tokens[i++], TOKEN_SEMICOLON, ";");
    assert_token(tokens[i++], TOKEN_KEYWORD, "return");
    assert_token(tokens[i++], TOKEN_INTEGER_LITERAL, "0");
    assert_token(tokens[i++], TOKEN_SEMICOLON, ";");
    assert_token(tokens[i++], TOKEN_RBRACE, "}");
    assert_token(tokens[i++], TOKEN_EOF, "");

    printf("Lexer test passed.\n");

    for (int j = 0; j < count; j++) {
        free(tokens[j].lexeme);
    }
    free(tokens);

    return 0;
}


