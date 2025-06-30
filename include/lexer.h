#ifndef LEXER_H
#define LEXER_H

#include "token.h"

// Returns the input C string source as an output token stream  
Token *tokenize(const char *source, int *count);

// Frees the input token stream
void free_tokens(Token *tokens, int count);

#endif
