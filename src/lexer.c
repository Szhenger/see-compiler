#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "lexer.h"

// === Internal Keyword Table ===
static const char *keywords[] = { "int", "return", "void" };

// === Private Helper: Check if word is a keyword ===
int is_keyword(const char *word) 
{
    for (size_t i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
        if (strcmp(word, keywords[i]) == 0) return 1;
    }
    return 0;
}

/*
 * Tokenization Strategy:
 * This lexer performs a single-pass, greedy scan over the input source.
 * It supports:
 * - Keywords: "int", "return", "void"
 * - Identifiers: alphanumeric words starting with a letter or underscore
 * - Integer literals: sequences of digits
 * - String literals: characters between double quotes
 * - Symbols: single characters like (, ), {, }, ;, =
 * - Whitespace is ignored; unknown characters are returned as TOKEN_UNKNOWN
 *
 * Tokens are dynamically allocated and stored in a resizable array.
 * Memory for each token's lexeme is also dynamically allocated.
 */

// === Public Helper: Read next token from input pointer ===
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

// === Public Function: Tokenize source into dynamic array of tokens ===
Token *tokenize(const char *source, int *count) 
{
    const char *input = source;
    int capacity = 18;
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

// Public Helper: Frees the token stream
void free_tokens(Token *tokens, int count) {
    if (!tokens) return;
    for (int i = 0; i < count; i++) {
        free(tokens[i].lexeme);
    }
    free(tokens);
}


