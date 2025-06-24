#ifndef LEXER_H
#define LEXER_H

// Include the set of tokens defined in token.c
#include "token.h"

// Tokenize source into dynamic array of tokens
Token *tokenize(const char *source, int *count); 

// Check if word is a keyword
int is_keyword(const char *word);

// Returns the next token in input stream
Token next_token(const char **input);

// Frees a dynamic array of tokens 
void free_tokens(Token *tokens, int count);


#endif
