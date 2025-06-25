#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "lexer.h"

// === Internal Keyword Table ===
static const char *keywords[] = {
    "int", "return", "void", "if", "else", "while", "for",
    "bool", "true", "false", "string", "include"
};

// === Internal Symbol Table ===
static const char *multi_char_symbols[] = {
    "==", "!=", "<=", ">=", "&&", "||"
};

// === Public Helper: Check if word is a keyword ===
int is_keyword(const char *word) {
    for (size_t i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
        if (strcmp(word, keywords[i]) == 0) return 1;
    }
    return 0;
}

// === Public Helper: Check for a multi-char symbol match ===
static const char *match_multi_char_symbol(const char *input) {
    for (size_t i = 0; i < sizeof(multi_char_symbols) / sizeof(multi_char_symbols[0]); i++) {
        size_t len = strlen(multi_char_symbols[i]);
        if (strncmp(input, multi_char_symbols[i], len) == 0) {
            return multi_char_symbols[i];
        }
    }
    return NULL;
}

// === Public Helper: Read next token ===
Token next_token(const char **input) {
    while (**input && isspace(**input)) (*input)++;
    if (**input == '\0') return (Token){ TOKEN_EOF, strdup(""), 0, 0 };

    // Handle comments
    if (**input == '/' && (*input)[1] == '/') {
        while (**input && **input != '\n') (*input)++;
        return next_token(input);
    }
    if (**input == '/' && (*input)[1] == '*') {
        (*input) += 2;
        while (**input && !(**input == '*' && (*input)[1] == '/')) (*input)++;
        if (**input) (*input) += 2;
        return next_token(input);
    }

    // Check for multi-char symbols
    const char *sym = match_multi_char_symbol(*input);
    if (sym) {
        (*input) += strlen(sym);
        return (Token){ TOKEN_SYMBOL, strdup(sym), 0, 0 };
    }

    // Check for single-char symbols
    char c = **input;
    if (strchr("(){}[];,=<>!+-*/%&|", c)) {
        (*input)++;
        char *lexeme = malloc(2); lexeme[0] = c; lexeme[1] = '\0';
        return (Token){ TOKEN_SYMBOL, lexeme, 0, 0 };
    }

    // Integer literal
    if (isdigit(c)) {
        char buffer[32]; int i = 0;
        while (isdigit(**input)) buffer[i++] = *(*input)++;
        buffer[i] = '\0';
        return (Token){ TOKEN_INTEGER_LITERAL, strdup(buffer), 0, 0 };
    }

    // String literal
    if (c == '"') {
        (*input)++;
        char buffer[256]; int i = 0;
        while (**input && **input != '"') {
            if (**input == '\\') buffer[i++] = *(*input)++; // escape char
            buffer[i++] = *(*input)++;
        }
        if (**input == '"') (*input)++; // skip closing quote
        buffer[i] = '\0';
        return (Token){ TOKEN_STRING_LITERAL, strdup(buffer), 0, 0 };
    }

    // Identifier or keyword
    if (isalpha(c) || c == '_') {
        char buffer[64]; int i = 0;
        while (isalnum(**input) || **input == '_') buffer[i++] = *(*input)++;
        buffer[i] = '\0';
        return (Token){
            is_keyword(buffer) ? TOKEN_KEYWORD : TOKEN_IDENTIFIER,
            strdup(buffer), 0, 0
        };
    }

    // Unknown token
    (*input)++;
    return (Token){ TOKEN_UNKNOWN, strdup("?"), 0, 0 };
}

// === Public Function: Tokenize entire source ===
Token *tokenize(const char *source, int *count) {
    const char *input = source;
    int capacity = 64;
    int size = 0;
    Token *tokens = malloc(sizeof(Token) * capacity);

    while (true) {
        Token t = next_token(&input);
        if (size >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, sizeof(Token) * capacity);
        }
        tokens[size++] = t;
        if (t.type == TOKEN_EOF) break;
    }

    *count = size;
    return tokens;
}

// === Public Function: Free token array ===
void free_tokens(Token *tokens, int count) {
    if (!tokens) return;
    for (int i = 0; i < count; i++) {
        free(tokens[i].lexeme);
    }
    free(tokens);
}
