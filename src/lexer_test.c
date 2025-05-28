#include <stdio.h>
#include "lexer.c"

int main(void) 
{
    const char *input = "int main(void) { return 0; }";
    Token tok;
    while ((tok = next_token(&input)).type != TOKEN_EOF) {
        printf("[%s, \"%s\"]\n", token_kind_to_string(tok.type), tok.lexeme);
        free(tok.lexeme);
    }
    return 0;
}
