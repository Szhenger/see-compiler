#ifndef TOKEN_H
#define TOKEN_H

typedef enum {
    TOKEN_KEYWORD,            

    TOKEN_IDENTIFIER,         

    TOKEN_INTEGER_LITERAL,    
    TOKEN_STRING_LITERAL,     
    TOKEN_CHAR_LITERAL,       

    TOKEN_LPAREN,            
    TOKEN_RPAREN,             
    TOKEN_LBRACE,             
    TOKEN_RBRACE,             
    TOKEN_LBRACKET,          
    TOKEN_RBRACKET,           
    TOKEN_SEMICOLON,          
    TOKEN_COMMA,              
    TOKEN_DOT,                

    TOKEN_ASSIGN,             
    TOKEN_PLUS,               
    TOKEN_MINUS,             
    TOKEN_STAR,               
    TOKEN_SLASH,              
    TOKEN_PERCENT,            
    TOKEN_INCREMENT,          
    TOKEN_DECREMENT,          

    TOKEN_EQUAL,              
    TOKEN_NOT_EQUAL,          
    TOKEN_LESS,               
    TOKEN_GREATER,            
    TOKEN_LESS_EQUAL,         
    TOKEN_GREATER_EQUAL,      

    TOKEN_AND,                
    TOKEN_OR,                 
    TOKEN_NOT,                

    TOKEN_BIT_AND,            
    TOKEN_BIT_OR,             
    TOKEN_BIT_XOR,            
    TOKEN_BIT_NOT,            
    TOKEN_LEFT_SHIFT,         
    TOKEN_RIGHT_SHIFT,        

    TOKEN_ARROW,              
    TOKEN_QUESTION,           
    TOKEN_COLON,              

    TOKEN_UNKNOWN,
    TOKEN_EOF
} TokenCategory;

typedef struct {
    TokenCategory category;
    char *lexeme;     
    int line;               
    int column;                
} Token;

const char *token_category_to_string(TokenCategory category);

#endif
