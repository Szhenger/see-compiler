#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "lexer.h"

// == Internal Keyword Table ==
static const char *keywords[] = {
    "int", "char", "bool", "string", "void", "return", 
    "if", "else", "while", "for", 
    "true", "false",
};

// == Internal Muti-Char Symbols Table ==
static const char *multi_char_symbols[] = {
    "==", "!=", "<=", ">=", "&&", "||", "++", "--", "->", "<<", ">>"
};

// == Private Helper: Checks whether input string word is a keyword ==
static int is_keyword(const char *word) 
{
    for (size_t i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
        if (strcmp(word, keywords[i]) == 0) return 1;
    }
    return 0;
}

// == Private Helper: Checks whether input string is a multi-char symbol ==
static const char *match_multi_char_symbol(const char *input) 
{
    for (size_t i = 0; i < sizeof(multi_char_symbols) / sizeof(multi_char_symbols[0]); i++) {
        size_t len = strlen(multi_char_symbols[i]);
        if (strncmp(input, multi_char_symbols[i], len) == 0) {
            return multi_char_symbols[i];
        }
    }
    return NULL;
}

// == Private Helper: Finds the associated single-char token category ==
static TokenCategory find_single_char_symbol_category(const char symbol) 
{
    switch (symbol) {
        case '(':           return TOKEN_LPAREN;
        case ')':           return TOKEN_RPAREN;
        case '{':           return TOKEN_LBRACE;
        case '}':           return TOKEN_RBRACE;
        case '[':           return TOKEN_LBRACKET;
        case ']':           return TOKEN_RBRACKET;
        case ',':           return TOKEN_COMMA;
        case ';':           return TOKEN_SEMICOLON;
        
        case '=':           return TOKEN_ASSIGN;
        case '+':           return TOKEN_PLUS;
        case '-':           return TOKEN_MINUS;
        case '*':           return TOKEN_STAR;
        case '/':           return TOKEN_SLASH;
        case '<':           return TOKEN_LESS;
        case '>':           return TOKEN_GREATER;
        case '!':           return TOKEN_NOT;
        
        case '&':           return TOKEN_BIT_AND;
        case '|':           return TOKEN_BIT_OR;
        case '^':           return TOKEN_BIT_XOR;
        case '~':           return TOKEN_BIT_NOT;
        
        case '.':           return TOKEN_DOT;
        case '?':           return TOKEN_QUESTION;
        case ':':           return TOKEN_COLON;
        
        default:            return TOKEN_UNKNOWN;
    }
}

// == Private Helper: Finds the associated multi-char token category == 
static TokenCategory find_multi_char_symbol_category(const char *symbol) 
{
    if (strcmp(symbol, "==") == 0) return TOKEN_EQUAL;
    if (strcmp(symbol, "!=") == 0) return TOKEN_NOT_EQUAL;
    if (strcmp(symbol, "<=") == 0) return TOKEN_LESS_EQUAL;
    if (strcmp(symbol, ">=") == 0) return TOKEN_GREATER_EQUAL;
    if (strcmp(symbol, "&&") == 0) return TOKEN_AND;
    if (strcmp(symbol, "||") == 0) return TOKEN_OR;
    if (strcmp(symbol, "++") == 0) return TOKEN_INCREMENT;
    if (strcmp(symbol, "--") == 0) return TOKEN_DECREMENT;
    if (strcmp(symbol, "->") == 0) return TOKEN_ARROW;
    if (strcmp(symbol, "<<") == 0) return TOKEN_LEFT_SHIFT;
    if (strcmp(symbol, ">>") == 0) return TOKEN_RIGHT_SHIFT;
    return TOKEN_UNKNOWN;
}

// == Private Helper: Generates the C substring input token ==
static Token next_token(const char **input) {
    // Skips whitespace
    while (**input && isspace(**input)) (*input)++;
    // Checks for NUL terminating value
    if (**input == '\0') return (Token){ TOKEN_EOF, strdup(""), 0, 0 };
    
    // Skips comments
    if (**input == '/' && *(*input + 1) == '/') {
        while (**input && **input != '\n') (*input)++;
        return next_token(input);
    }
    if (**input == '/' && *(*input + 1) == '*') {
        (*input) += 2;
        while (**input && !(**input == '*' && (*input)[1] == '/')) (*input)++;
        if (**input) (*input) += 2;
        return next_token(input);
    }

    // Checks for muti-char symbol
    const char *symbol = match_multi_char_symbol(*input);
    if (symbol) {
        (*input) += strlen(symbol);
        return (Token){ find_multi_char_symbol_category(symbol), strdup(symbol), 0, 0 };
    }

    char c = **input;

    // Checks for single-char symbol
    if (strchr("(){}[];,=<>!+-*/%&|", c)) {
        (*input)++;
        char *lexeme = malloc(2); lexeme[0] = c; lexeme[1] = '\0';
        return (Token){ find_single_char_symbol_category(c), lexeme, 0, 0 };
    }

    // Checks for integer literal
    if (isdigit(c)) {
        char buffer[32]; int i = 0;
        while (isdigit(**input)) buffer[i++] = *(*input)++;
        buffer[i] = '\0';
        return (Token){ TOKEN_INTEGER_LITERAL, strdup(buffer), 0, 0 };
    }

    // Checks for string literal
    if (c == '"') {
        (*input)++;
        char buffer[256]; int i = 0;
        while (**input && **input != '"') {
            if (**input == '\\') buffer[i++] = *(*input)++;
            buffer[i++] = *(*input)++;
        }
        if (**input == '"') (*input)++;
        buffer[i] = '\0';
        return (Token){ TOKEN_STRING_LITERAL, strdup(buffer), 0, 0 };
    }
    
    // Checks for char literal
    if (c == '\'') {
        (*input)++;
        char ch = **input;
        (*input)++;
        if (**input == '\'') (*input)++;
        char buffer[2] = { ch, '\0' };
        return (Token){ TOKEN_CHAR_LITERAL, strdup(buffer), 0, 0 };
    }
    
    // Checks for keyword
    if (isalpha(c) || c == '_') {
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

// == Public Function: Returns the associated dynamic array of tokens ==
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
        if (t.category == TOKEN_EOF) break;
    }

    *count = size;
    return tokens;
}

// == Public Function: Frees the input dynamic array of tokens ==  
void free_tokens(Token *tokens, int count) {
    if (!tokens) return;
    for (int i = 0; i < count; i++) {
        free(tokens[i].lexeme);
    }
    free(tokens);
}
