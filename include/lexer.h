#ifndef LEXER_H
#define LEXER_H

// Include the set of tokens defined in token.c
#include "token.h"

// Returns the next token in input stream
Token next_token(const char **input);

#endif
