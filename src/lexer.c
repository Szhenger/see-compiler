#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "token.h"

static const char *keywords[] = { "int", "return" };

static int is_keyword(const char *word) 
{
    for (size_t i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
        if (strcmp(word, keywords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

Token next_token(const char **input) 
{
    // Skip whitespace
    while (isspace(**input)) {
        (*input)++;
    }

    if (**input == '\0') {
        return (Token){ TOKEN_EOF, strdup(""), 0, 0 };
    }

    // Symbol (1-char)
    char c = **input;
    if (strchr("(){};=", c)) {
        char sym[2] = { c, '\0' };
        (*input)++;
        return (Token){ TOKEN_SYMBOL, strdup(sym), 0, 0 };
    }

    // Integer literal
    if (isdigit(c)) {
        char buffer[32];
        int i = 0;
        while (isdigit(**input)) {
            buffer[i++] = **input;
            (*input)++;
        }
        buffer[i] = '\0';
        return (Token){ TOKEN_INTEGER_LITERAL, strdup(buffer), 0, 0 };
    }

    // Identifier or keyword
    if (isalpha(c)) {
        char buffer[64];
        int i = 0;
        while (isalnum(**input)) {
            buffer[i++] = **input;
            (*input)++;
        }
        buffer[i] = '\0';
        if (is_keyword(buffer)) {
            return (Token){ TOKEN_KEYWORD, strdup(buffer), 0, 0 };
        } else {
            return (Token){ TOKEN_IDENTIFIER, strdup(buffer), 0, 0 };
        }
    }

    // Fallback
    (*input)++;
    return (Token){ TOKEN_UNKNOWN, strdup("?"), 0, 0 };
}
