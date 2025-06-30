#ifndef TOKEN_H
#define TOKEN_H

// Enumerates the categories of supported tokens i.e. C/C++ syntax
typedef enum {
    // Keywords
    TOKEN_KEYWORD,

    // Function and Variable Names
    TOKEN_IDENTIFIER,

    // Literals
    TOKEN_INTEGER_LITERAL,    
    TOKEN_STRING_LITERAL,     
    TOKEN_CHAR_LITERAL,

    // Symbols
    TOKEN_LPAREN,            
    TOKEN_RPAREN,             
    TOKEN_LBRACE,             
    TOKEN_RBRACE,             
    TOKEN_LBRACKET,          
    TOKEN_RBRACKET,           
    TOKEN_SEMICOLON,          
    TOKEN_COMMA,              
    TOKEN_DOT,       

    // Binary Operators
    TOKEN_ASSIGN,             
    TOKEN_PLUS,               
    TOKEN_MINUS,             
    TOKEN_STAR,               
    TOKEN_SLASH,              
    TOKEN_PERCENT,            
    TOKEN_INCREMENT,          
    TOKEN_DECREMENT,    

    // Comparison Operators
    TOKEN_EQUAL,              
    TOKEN_NOT_EQUAL,          
    TOKEN_LESS,               
    TOKEN_GREATER,            
    TOKEN_LESS_EQUAL,         
    TOKEN_GREATER_EQUAL, 

    // Logical Operators
    TOKEN_AND,                
    TOKEN_OR,                 
    TOKEN_NOT, 

    // Bitwise Operators
    TOKEN_BIT_AND,            
    TOKEN_BIT_OR,             
    TOKEN_BIT_XOR,            
    TOKEN_BIT_NOT,            
    TOKEN_LEFT_SHIFT,         
    TOKEN_RIGHT_SHIFT,  

    // Special Operators 
    TOKEN_ARROW,              
    TOKEN_QUESTION,           
    TOKEN_COLON,       

    // Sentient Values
    TOKEN_UNKNOWN,
    TOKEN_EOF
} TokenCategory;

// Defines the structure of a token i.e. C/C++ data and instructions
typedef struct {
    TokenCategory category; // Category Field
    char *lexeme;           // Literal Field     
    int line;               // Line Number Field (Optional)               
    int column;             // Column Number Field (Optional) 
} Token;

// Returns the inputted token category as a string for debugging 
const char *token_category_to_string(TokenCategory category);

#endif
