#ifndef TOKEN_H
#define TOKEN_H

// Enumerates the types of tokens the lexer can recognize
typedef enum {
    TOKEN_KEYWORD,          // Keywords like "int", "return"
    TOKEN_IDENTIFIER,       // User-defined names like "main", "printf"
    TOKEN_INTEGER_LITERAL,  // Integers like 0, 42
    TOKEN_STRING_LITERAL,   // String constants like "Hello, world!\n"
    TOKEN_SYMBOL,           // Symbols like (, ), {, }, ;
    TOKEN_UNKNOWN,          // Unrecognized token
    TOKEN_EOF               // End of input    
} TokenType;

// Represents a single token in the token stream
typedef struct {
    TokenType type;     // The type of token
    char *lexeme;       // A string copy of the token's actual text
    int line;           // Line number in source (optional but helpful)
    int column;         // Column number in source (optional but helpful)
} Token;

// Returns a string name for a TokenType enum (for debugging or printing)
const char *token_type_to_string(TokenType type);

#endif // TOKEN_H
