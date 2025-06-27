#ifndef TOKEN_H
#define TOKEN_H

// Enumerates the types of tokens the lexer can recognize
typedef enum {
    // Keywords
    TOKEN_KEYWORD,            // int, return, void, if, else, while, for, break, continue, bool, true, false

    // Identifiers
    TOKEN_IDENTIFIER,         // Function names, variable names

    // Literals
    TOKEN_INTEGER_LITERAL,    // 0, 42, etc.
    TOKEN_STRING_LITERAL,     // "hello"
    TOKEN_CHAR_LITERAL,       // 'a'

    // Symbols
    TOKEN_LPAREN,             // (
    TOKEN_RPAREN,             // )
    TOKEN_LBRACE,             // {
    TOKEN_RBRACE,             // }
    TOKEN_LBRACKET,           // [
    TOKEN_RBRACKET,           // ]
    TOKEN_SEMICOLON,          // ;
    TOKEN_COMMA,              // ,
    TOKEN_DOT,                // .

    // Operators
    TOKEN_ASSIGN,             // =
    TOKEN_PLUS,               // +
    TOKEN_MINUS,              // -
    TOKEN_STAR,               // *
    TOKEN_SLASH,              // /
    TOKEN_PERCENT,            // %
    TOKEN_INCREMENT,          // ++
    TOKEN_DECREMENT,          // --

    // Comparison
    TOKEN_EQUAL,              // ==
    TOKEN_NOT_EQUAL,          // !=
    TOKEN_LESS,               // <
    TOKEN_GREATER,            // >
    TOKEN_LESS_EQUAL,         // <=
    TOKEN_GREATER_EQUAL,      // >=

    // Logical
    TOKEN_AND,                // &&
    TOKEN_OR,                 // ||
    TOKEN_NOT,                // !

    // Bitwise
    TOKEN_BIT_AND,            // &
    TOKEN_BIT_OR,             // |
    TOKEN_BIT_XOR,            // ^
    TOKEN_BIT_NOT,            // ~
    TOKEN_LEFT_SHIFT,         // <<
    TOKEN_RIGHT_SHIFT,        // >>

    // Special
    TOKEN_ARROW,              // ->
    TOKEN_QUESTION,           // ?
    TOKEN_COLON,              // :

    // Miscellaneous 
    TOKEN_UNKNOWN,
    TOKEN_EOF
} TokenCategory;

// Represents a immutable token in the token stream
typedef struct {
    TokenCategory category;     // The category of token
    char *lexeme;               // A string copy of the token's actual text and heap-allocated, so must be freed by the token consumer
    int line;                   // Line number in source (optional but helpful)
    int column;                 // Column number in source (optional but helpful)
} Token;

// Returns a string name for a TokenType enum (for debugging or printing)
const char *token_category_to_string(TokenCategory category);

#endif // TOKEN_H
