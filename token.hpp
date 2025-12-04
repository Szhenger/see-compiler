#ifndef TOKEN_HPP
#define TOKEN_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace token {

struct SourceLocation {
    std::uint32_t line;
    std::uint32_t column;
};

enum class TokenCategory : std::uint8_t {
    Primitive,
    Container,
    Keyword,
    Operator,
    Punctuation,
    Literal,
    Identifier,
    Preprocessor,
    EndOfFile,
    Unknown
};

enum class PrimitiveKind : std::uint8_t {
    Void,
    Char,
    Short,
    Int,
    Long,
    Float,
    Double,
    Bool,       
    Pointer,
    Array
};

enum class ContainerKind : std::uint8_t {
    String,
    Vector,
    LinkedList,
    Stack,
    Queue
};

enum class KeywordKind : std::uint16_t {
    // C
    Auto, Break, Case, Const, Continue, Default, Do, Else, Enum, Extern,
    For, Goto, If, Inline, Register, Restrict, Return, Signed, Sizeof,
    Static, Struct, Switch, Typedef, Union, Unsigned, Volatile, While,
    VoidKw, IntKw, FloatKw, DoubleKw, CharKw, LongKw, ShortKw,

    // C++
    Alignas, Alignof, BoolKw, Class, Constexpr, ConstCast, Decltype,
    Delete, DynamicCast, Explicit, Export, FalseKw, Friend, Mutable,
    Namespace, New, Noexcept, Nullptr, Operator, Private, Protected,
    Public, ReinterpretCast, StaticAssert, StaticCast, Template,
    This, ThreadLocal, Throw, TrueKw, Try, Typeid, Typename, Using,
    Virtual
};

enum class OperatorKind : std::uint16_t {
    Plus, Minus, Star, Slash, Percent,
    LogicalAnd, LogicalOr, LogicalNot,
    BitAnd, BitOr, BitXor, BitNot,
    ShiftLeft, ShiftRight,
    Assign, PlusAssign, MinusAssign, MulAssign, DivAssign, ModAssign,
    AndAssign, OrAssign, XorAssign, ShlAssign, ShrAssign,
    Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    Increment, Decrement,
    Arrow, Dot,
    Scope,        // ::
    MemberPtr,    // .* and ->*
    SizeofOp, AlignofOp, TypeidOp
};

enum class PunctuationKind : std::uint8_t {
    LParen, RParen,
    LBracket, RBracket,
    LBrace, RBrace,
    Semicolon,
    Colon,
    Comma,
    Ellipsis,
    Question,
    TemplateLT, TemplateGT
};

enum class LiteralKind : std::uint8_t {
    Integer,
    Float,
    Char,
    String,
    Bool,
    Null
};

enum class PreprocessorKind : std::uint8_t {
    Define, Undef,
    If, Ifdef, Ifndef, Elif, Else, Endif,
    Line, Error, Pragma, Include,
    MacroIdentifier,
    MacroNumber,
    MacroString,
    MacroChar,
    Paste,      // ##
    Stringize   // #
};

struct PrimitiveToken {
    PrimitiveKind kind;
    bool is_signed;
    bool is_const;
    bool is_volatile;
    std::uint32_t array_size; // Only valid when kind == Array
};

struct ContainerToken {
    ContainerKind kind;
};

struct LiteralData {
    LiteralKind kind;
    union {
        std::int64_t int_val;
        double float_val;
        char char_val;
    };
    std::string string_val; // Used for string literals
};

struct Token {
    TokenCategory category;
    SourceLocation location;
    std::string lexeme;
    union {
        PrimitiveToken primitive;
        ContainerToken container;
        KeywordKind keyword;
        OperatorKind op;
        PunctuationKind punct;
        LiteralData literal;
        PreprocessorKind pp;
    };
};

Token make_primitive_token(PrimitiveKind kind,
                           bool is_signed,
                           bool is_const,
                           bool is_volatile,
                           std::uint32_t array_size,
                           const SourceLocation& loc,
                           const std::string& lexeme);

Token make_container_token(ContainerKind kind,
                           const SourceLocation& loc,
                           const std::string& lexeme);

Token make_keyword_token(KeywordKind kind,
                         const SourceLocation& loc,
                         const std::string& lexeme);

Token make_operator_token(OperatorKind kind,
                          const SourceLocation& loc,
                          const std::string& lexeme);

Token make_punctuation_token(PunctuationKind kind,
                             const SourceLocation& loc,
                             const std::string& lexeme);

Token make_literal_int(std::int64_t value,
                       const SourceLocation& loc,
                       const std::string& lexeme);

Token make_literal_float(double value,
                         const SourceLocation& loc,
                         const std::string& lexeme);

Token make_literal_char(char value,
                        const SourceLocation& loc,
                        const std::string& lexeme);

Token make_literal_string(const std::string& value,
                          const SourceLocation& loc,
                          const std::string& lexeme);

Token make_literal_bool(bool value,
                        const SourceLocation& loc,
                        const std::string& lexeme);

Token make_literal_null(const SourceLocation& loc,
                        const std::string& lexeme);

Token make_identifier_token(const SourceLocation& loc,
                            const std::string& lexeme);

Token make_preprocessor_token(PreprocessorKind kind,
                              const SourceLocation& loc,
                              const std::string& lexeme);

void destroy_token(Token& t);

Token copy_token(const Token& t);

Token move_token(Token&& t);

const char* primitive_to_string(PrimitiveKind kind);

const char* container_to_string(ContainerKind kind);

const char* keyword_to_string(KeywordKind kind);

const char* operator_to_string(OperatorKind kind);

const char* punctuation_to_string(PunctuationKind kind);

const char* literal_to_string(LiteralKind kind);

const char* preprocessor_to_string(PreprocessorKind kind);

const char* token_category_to_string(TokenCategory cat);

std::string token_to_string(const Token& t);

void print_token(const Token& t);

bool lookup_keyword(const std::string& text, KeywordKind& out_kind);

bool lookup_operator(const std::string& text, OperatorKind& out_kind);

bool lookup_punctuation(const std::string& text, PunctuationKind& out_kind);

bool is_primitive(const Token& t);

bool is_container(const Token& t);

bool is_keyword(const Token& t);

bool is_operator(const Token& t);

bool is_punctuation(const Token& t);

bool is_literal(const Token& t);

bool is_identifier(const Token& t);

bool is_preprocessor(const Token& t);

bool is_type_keyword(const Token& t);

bool is_unary_operator(const Token& t);

bool is_binary_operator(const Token& t);

}

#endif // TOKEN_HPP

