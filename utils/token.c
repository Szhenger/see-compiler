#include "token.h"

// Maps token type to string variables for debugging
const char *token_type_to_string(TokenType type) {
    switch (type) {
        case TOKEN_EOF:              return "TOKEN_EOF";
        case TOKEN_KEYWORD:          return "TOKEN_KEYWORD";
        case TOKEN_IDENTIFIER:       return "TOKEN_IDENTIFIER";
        case TOKEN_INTEGER_LITERAL:  return "TOKEN_INTEGER_LITERAL";
        case TOKEN_STRING_LITERAL:   return "TOKEN_STRING_LITERAL";
        case TOKEN_SYMBOL:           return "TOKEN_SYMBOL";
        case TOKEN_UNKNOWN:          return "TOKEN_UNKNOWN";
        default:                     return "UNKNOWN_TOKEN_KIND";
    }
}
