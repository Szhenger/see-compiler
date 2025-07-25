#include <stdio.h>
#include "token.h"

// == Public Function: Returns strings of token categories == 
const char *token_category_to_string(TokenCategory category) {
    switch (category) {
        // Data 
        case TOKEN_KEYWORD:          return "TOKEN_KEYWORD";
        case TOKEN_IDENTIFIER:       return "TOKEN_IDENTIFIER";
        case TOKEN_INTEGER_LITERAL:  return "TOKEN_INTEGER_LITERAL";
        case TOKEN_STRING_LITERAL:   return "TOKEN_STRING_LITERAL";
        case TOKEN_CHAR_LITERAL:     return "TOKEN_CHAR_LITERAL";

        // Symbols
        case TOKEN_LPAREN:           return "TOKEN_LPAREN";
        case TOKEN_RPAREN:           return "TOKEN_RPAREN";
        case TOKEN_LBRACE:           return "TOKEN_LBRACE";
        case TOKEN_RBRACE:           return "TOKEN_RBRACE";
        case TOKEN_LBRACKET:         return "TOKEN_LBRACKET";
        case TOKEN_RBRACKET:         return "TOKEN_RBRACKET";
        case TOKEN_COMMA:            return "TOKEN_COMMA";
        case TOKEN_SEMICOLON:        return "TOKEN_SEMICOLON";

        // Mathematical Operators 
        case TOKEN_ASSIGN:           return "TOKEN_ASSIGN";
        case TOKEN_PLUS:             return "TOKEN_PLUS";
        case TOKEN_MINUS:            return "TOKEN_MINUS";
        case TOKEN_STAR:             return "TOKEN_STAR";
        case TOKEN_SLASH:            return "TOKEN_SLASH";
        case TOKEN_LESS:             return "TOKEN_LESS";
        case TOKEN_GREATER:          return "TOKEN_GREATER";
        case TOKEN_EQUAL:            return "TOKEN_EQUAL";
        case TOKEN_NOT_EQUAL:        return "TOKEN_NOT_EQUAL";
        case TOKEN_LESS_EQUAL:       return "TOKEN_LESS_EQUAL";
        case TOKEN_GREATER_EQUAL:    return "TOKEN_GREATER_EQUAL";
        case TOKEN_AND:              return "TOKEN_AND";
        case TOKEN_OR:               return "TOKEN_OR";
        case TOKEN_NOT:              return "TOKEN_NOT";

        // Sentinal Values 
        case TOKEN_EOF:              return "TOKEN_EOF";
        case TOKEN_UNKNOWN:          return "TOKEN_UNKNOWN";
        
        default:                     return "UNKNOWN_TOKEN_KIND";
    }
}

// == Public Function: Prints token stream to standard output == 
void print_tokens(const Token *tokens, int count) 
{
    printf("== Token Stream ==\n");
    for (int i = 0; i < count; i++) {
        printf("[%d] Category: %s | Lexeme: \"%s\"\n", 
               i, token_category_to_string(tokens[i].category), tokens[i].lexeme);
    }
}
