#include <stdio.h>
#include <stdlib.h>
#include "lexer.h"
#include "token.h"

int main(void) 
{
    const char *source =
        "int main(void) {\n"
        "    printf(\"Hello, world!\\n\");\n"
        "    return 0;\n"
        "}";

    const char *input = source;
    Token t;

    do {
        t = next_token(&input);
        printf("[%s, \"%s\"]\n", token_kind_to_string(t.kind), t.lexeme);
        free(t.lexeme);
    } while (t.type != TOKEN_EOF);

    return 0;
}
