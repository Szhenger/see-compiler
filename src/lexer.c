#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include "lexer.h"

static const char *keywords[] = { "int", "return", "void" };

// Helper function to verify keywords 
static int is_keyword(const char *word) 
{
    for (size_t i = 0; i < sizeof(keywords)/sizeof(keywords[0]); i++) {
        if (strcmp(word, keywords[i]) == 0) return 1;
    }
    return 0;
}

// Get token obtained from input pointer
Token next_token(const char **input) 
{
    // Increment on whitespaces
    while (**input && isspace(**input)) {
        (*input)++;
    }

    // Return NUL token on NUL character   
    if (**input == '\0') {
        return (Token){ TOKEN_EOF, strdup(""), 0, 0 };
    }

    // Get the char literal deferenced by input
    char c = **input;

    // See if c is bracket character
    if (strchr("(){};=", c)) {
        (*input)++;
        char *lexeme = malloc(2);
        lexeme[0] = c;
        lexeme[1] = '\0';
        return (Token){ TOKEN_SYMBOL, lexeme, 0, 0 };
    }

    // See if c is a integer literal
    if (isdigit(c)) {
        char buffer[32];
        int i = 0;
        while (isdigit(**input)) buffer[i++] = *(*input)++;
        buffer[i] = '\0';
        return (Token){ TOKEN_INTEGER_LITERAL, strdup(buffer), 0, 0 };
    }

    // See if c is a string literal
    if (c == '"') {
        (*input)++;
        char buffer[256];
        int i = 0;
        while (**input && **input != '"') buffer[i++] = *(*input)++;
        (*input)++; // skip closing quote
        buffer[i] = '\0';
        return (Token){ TOKEN_STRING_LITERAL, strdup(buffer), 0, 0 };
    }

    // See if c is either a keyword or a identifier
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

