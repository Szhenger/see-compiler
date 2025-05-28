#include <stdio.h>
#include <stdlib.h>
#include "lexer.h"
#include "token.h"

int main(void) 
{
    const char *source = "int main(void) { return 0; }";
    const char *input = source;
    
    Token t;
    do {
        t = next_token(&input);
        printf("[%s, \"%s\"]\n", token_type_to_string(t.type), t.lexeme);
        free(t.lexeme);
    } while (t.type != TOKEN_EOF);

    return 0;
}

