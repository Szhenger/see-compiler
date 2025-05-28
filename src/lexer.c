#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include "lexer.h"

static const char *keywords[] = { "int", "return", "void" };

static int is_keyword(const char *word) 
{
    for (size_t i = 0; i < sizeof(keywords)/sizeof(keywords[0]); i++) {
        if (strcmp(word, keywords[i]) == 0) return 1;
    }
    return 0;
}

Token next_token(const char **input) 
{
    while (**input && isspace(**input)) {
        (*input)++;
    }

    if (**input == '\0') {
        return (Token){ TOKEN_EOF, strdup(""), 0, 0 };
    }

    char c = **input;

    if (strchr("(){};=", c)) {
        (*input)++;
        char *lexeme = malloc(2);
        lexeme[0] = c;
        lexeme[1] = '\0';
        return (Token){ TOKEN_SYMBOL, lexeme, 0, 0 };
    }

    if (isdigit(c)) {
        char buffer[32];
        int i = 0;
        while (isdigit(**input)) buffer[i++] = *(*input)++;
        buffer[i] = '\0';
        return (Token){ TOKEN_INTEGER_LITERAL, strdup(buffer), 0, 0 };
    }

    if (c == '"') {
        (*input)++;
        char buffer[256];
        int i = 0;
        while (**input && **input != '"') buffer[i++] = *(*input)++;
        (*input)++; // skip closing quote
        buffer[i] = '\0';
        return (Token){ TOKEN_STRING_LITERAL, strdup(buffer), 0, 0 };
    }

    if (isalpha(c)) {
        char buffer[64];
        int i = 0;
        while (isalnum(**input) || **input == '_') buffer[i++] = *(*input)++;
        buffer[i] = '\0';
        if (is_keyword(buffer)) {
            return (Token){ TOKEN_KEYWORD, strdup(buffer), 0, 0 };
        } else {
            return (Token){ TOKEN_IDENTIFIER, strdup(buffer), 0, 0 };
        }
    }

    (*input)++;
    return (Token){ TOKEN_UNKNOWN, strdup("?"), 0, 0 };
}

