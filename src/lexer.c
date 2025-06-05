#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "lexer.h"

static const char *keywords[] = { "int", "return", "void" };

// Helper function prototype
Token next_token(const char **input);

// Public: tokenize source into dynamic array of tokens
Token *tokenize(const char *source, int *count) 
{
    const char *input = source;
    int capacity = 64;
    int size = 0;
    Token *tokens = malloc(sizeof(Token) * capacity);

    while (1) {
        Token t = next_token(&input);
        if (size >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, sizeof(Token) * capacity);
            // TODO (Error handling)
        }
        tokens[size++] = t;
        if (t.type == TOKEN_EOF) break;
    }

    *count = size;
    return tokens;
}

// Private: check if word is a keyword
int is_keyword(const char *word) 
{
    for (size_t i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
        if (strcmp(word, keywords[i]) == 0) return 1;
    }
    return 0;
}

// Public: read next token from input pointer
Token next_token(const char **input) 
{
    while (**input && isspace(**input)) (*input)++;
    if (**input == '\0') return (Token){ TOKEN_EOF, strdup(""), 0, 0 };

    char c = **input;

    if (strchr("(){};=", c)) {
        (*input)++;
        char *lexeme = malloc(2); lexeme[0] = c; lexeme[1] = '\0';
        return (Token){ TOKEN_SYMBOL, lexeme, 0, 0 };
    }

    if (isdigit(c)) {
        char buffer[32]; int i = 0;
        while (isdigit(**input)) buffer[i++] = *(*input)++;
        buffer[i] = '\0';
        return (Token){ TOKEN_INTEGER_LITERAL, strdup(buffer), 0, 0 };
    }

    if (c == '"') {
        (*input)++;
        char buffer[256]; int i = 0;
        while (**input && **input != '"') buffer[i++] = *(*input)++;
        (*input)++;
        buffer[i] = '\0';
        return (Token){ TOKEN_STRING_LITERAL, strdup(buffer), 0, 0 };
    }

    if (isalpha(c)) {
        char buffer[64]; int i = 0;
        while (isalnum(**input) || **input == '_') buffer[i++] = *(*input)++;
        buffer[i] = '\0';
        return (Token){
            is_keyword(buffer) ? TOKEN_KEYWORD : TOKEN_IDENTIFIER,
            strdup(buffer), 0, 0
        };
    }

    (*input)++;
    return (Token){ TOKEN_UNKNOWN, strdup("?"), 0, 0 };
}


