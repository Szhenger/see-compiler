#ifndef TOKEN_HPP
#define TOKEN_HPP

#include <cstdint>
#include <string>

typedef struct SourceLocation {
    uint32_t line;        // line number in file (1-based)
    uint32_t column;      // column number (1-based)
    uint32_t file_id;     // index for file table
    uint32_t offset;      // byte offset in source
} SourceLocation;

typedef enum TokenKind {
    TK_IDENTIFIER,        // variable/function names
    TK_KEYWORD,           // e.g., int, return, for
    TK_LITERAL,           // numeric/char/string constants
    TK_OPERATOR,          // +, -, *, ->, etc.
    TK_SEPARATOR,         // ;, (, ), {, }
    TK_TYPE_PRIMITIVE,    // primitive type token
    TK_TYPE_CONTAINER     // compound type token
} TokenKind;

typedef enum PrimitiveTokenKind {
    // C11 core primitives
    PT_CHAR,
    PT_VOID,
    PT_SHORT,
    PT_INT,
    PT_LONG,
    PT_FLOAT,
    PT_DOUBLE,

    // C++17 additions
    PT_BOOL,
    PT_STRING
} PrimitiveTokenKind;

typedef enum ContainerTokenKind {
    // C11 containers
    CT_ARRAY,
    CT_POINTER,

    // C++17 containers (minimal set)
    CT_VECTOR,
    CT_LIST,
    CT_STACK,
    CT_QUEUE
} ContainerTokenKind;

typedef union LiteralValue {
    int64_t      i64;     // integer literals
    double       f64;     // floating literals
    char         c;       // char constants
    const char  *str;     // string literals
} LiteralValue;

typedef struct PrimitiveToken {
    PrimitiveTokenKind kind;         // which primitive type
    
    const char         *lexeme;      // raw type lexeme (e.g., "int")
    uint32_t           qualifiers;   // const/volatile/unsigned/etc.
    
    SourceLocation     loc;          // location metadata
} PrimitiveToken;

typedef struct ContainerToken {
    ContainerTokenKind kind;         // which container

    PrimitiveToken     *data_type;     // for primitive children
    ContainerToken     *sub_container; // for nested containers

    uint64_t            size;
    SourceLocation      loc;
} ContainerToken;

typedef struct Token {
    TokenKind      token_kind;    // category of token
    const char    *lexeme;        // raw lexeme text
    SourceLocation loc;           // location in source

    LiteralValue   literal;

    PrimitiveToken *primitive_token;
    ContainerToken *container_token;
} Token;

#endif // TOKEN_HPP

