#ifndef LEXER_H
#define LEXER_H

#include "token.h"

// Tokenize the entire source into a dynamic array of tokens.
// Sets `count` to the number of tokens produced.
// The returned array must be freed by the caller using `free_tokens`.
Token *tokenize(const char *source, int *count);

// Determine whether a given word is a C keyword.
// Returns 1 if the word is a keyword (e.g., "int", "return", "if", etc.), 0 otherwise.
// int is_keyword(const char *word);

// Extracts the next token from the input stream.
// Advances the `input` pointer accordingly.
// Returns a Token with type, lexeme, and position info.
// Token next_token(const char **input);

// Frees a token array allocated by `tokenize`.
void free_tokens(Token *tokens, int count);

#endif // LEXER_H
