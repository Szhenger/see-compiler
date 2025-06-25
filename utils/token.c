#include "token.h"

// Maps token category to string representation for debugging
const char *token_category_to_string(TokenCategory category) {
    switch (type) {
        case TOKEN_KEYWORD:          return "TOKEN_KEYWORD";
        case TOKEN_IDENTIFIER:       return "TOKEN_IDENTIFIER";
        case TOKEN_INTEGER_LITERAL:  return "TOKEN_INTEGER_LITERAL";
        case TOKEN_STRING_LITERAL:   return "TOKEN_STRING_LITERAL";

        case TOKEN_LPAREN:           return "TOKEN_LPAREN";
        case TOKEN_RPAREN:           return "TOKEN_RPAREN";
        case TOKEN_LBRACE:           return "TOKEN_LBRACE";
        case TOKEN_RBRACE:           return "TOKEN_RBRACE";
        case TOKEN_COMMA:            return "TOKEN_COMMA";
        case TOKEN_SEMICOLON:        return "TOKEN_SEMICOLON";
        case TOKEN_ASSIGN:           return "TOKEN_ASSIGN";
        case TOKEN_PLUS:             return "TOKEN_PLUS";
        case TOKEN_MINUS:            return "TOKEN_MINUS";
        case TOKEN_STAR:             return "TOKEN_STAR";
        case TOKEN_SLASH:            return "TOKEN_SLASH";
        case TOKEN_LT:               return "TOKEN_LT";
        case TOKEN_GT:               return "TOKEN_GT";
        case TOKEN_EQ:               return "TOKEN_EQ";
        case TOKEN_NEQ:              return "TOKEN_NEQ";
        case TOKEN_LEQ:              return "TOKEN_LEQ";
        case TOKEN_GEQ:              return "TOKEN_GEQ";
        case TOKEN_AND:              return "TOKEN_AND";
        case TOKEN_OR:               return "TOKEN_OR";
        case TOKEN_NOT:              return "TOKEN_NOT";

        case TOKEN_EOF:              return "TOKEN_EOF";
        case TOKEN_UNKNOWN:          return "TOKEN_UNKNOWN";

        default:                     return "UNKNOWN_TOKEN_KIND";
    }
}

