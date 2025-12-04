#include "token.hpp"

#include <new>
#include <iostream>
#include <sstream>

namespace token {

static const std::vector<std::pair<std::string, KeywordKind>>& keyword_table() {
    static std::vector<std::pair<std::string, KeywordKind>> tbl = {
        // C
        {"auto", KeywordKind::Auto}, {"break", KeywordKind::Break}, {"case", KeywordKind::Case},
        {"const", KeywordKind::Const}, {"continue", KeywordKind::Continue}, {"default", KeywordKind::Default},
        {"do", KeywordKind::Do}, {"else", KeywordKind::Else}, {"enum", KeywordKind::Enum},
        {"extern", KeywordKind::Extern}, {"for", KeywordKind::For}, {"goto", KeywordKind::Goto},
        {"if", KeywordKind::If}, {"inline", KeywordKind::Inline}, {"register", KeywordKind::Register},
        {"restrict", KeywordKind::Restrict}, {"return", KeywordKind::Return}, {"signed", KeywordKind::Signed},
        {"sizeof", KeywordKind::Sizeof}, {"static", KeywordKind::Static}, {"struct", KeywordKind::Struct},
        {"switch", KeywordKind::Switch}, {"typedef", KeywordKind::Typedef}, {"union", KeywordKind::Union},
        {"unsigned", KeywordKind::Unsigned}, {"volatile", KeywordKind::Volatile}, {"while", KeywordKind::While},
        {"void", KeywordKind::VoidKw}, {"int", KeywordKind::IntKw}, {"float", KeywordKind::FloatKw},
        {"double", KeywordKind::DoubleKw}, {"char", KeywordKind::CharKw}, {"long", KeywordKind::LongKw},
        {"short", KeywordKind::ShortKw},

        // C++
        {"alignas", KeywordKind::Alignas}, {"alignof", KeywordKind::Alignof}, {"bool", KeywordKind::BoolKw},
        {"class", KeywordKind::Class}, {"constexpr", KeywordKind::Constexpr}, {"const_cast", KeywordKind::ConstCast},
        {"decltype", KeywordKind::Decltype}, {"delete", KeywordKind::Delete}, {"dynamic_cast", KeywordKind::DynamicCast},
        {"explicit", KeywordKind::Explicit}, {"export", KeywordKind::Export}, {"false", KeywordKind::FalseKw},
        {"friend", KeywordKind::Friend}, {"mutable", KeywordKind::Mutable}, {"namespace", KeywordKind::Namespace},
        {"new", KeywordKind::New}, {"noexcept", KeywordKind::Noexcept}, {"nullptr", KeywordKind::Nullptr},
        {"operator", KeywordKind::Operator}, {"private", KeywordKind::Private}, {"protected", KeywordKind::Protected},
        {"public", KeywordKind::Public}, {"reinterpret_cast", KeywordKind::ReinterpretCast},
        {"static_assert", KeywordKind::StaticAssert}, {"static_cast", KeywordKind::StaticCast},
        {"template", KeywordKind::Template}, {"this", KeywordKind::This}, {"thread_local", KeywordKind::ThreadLocal},
        {"throw", KeywordKind::Throw}, {"true", KeywordKind::TrueKw}, {"try", KeywordKind::Try},
        {"typeid", KeywordKind::Typeid}, {"typename", KeywordKind::Typename}, {"using", KeywordKind::Using},
        {"virtual", KeywordKind::Virtual}
    };
    return tbl;
}

static const std::vector<std::pair<std::string, OperatorKind>>& operator_table() {
    static std::vector<std::pair<std::string, OperatorKind>> tbl = {
        {"+", OperatorKind::Plus}, {"-", OperatorKind::Minus}, {"*", OperatorKind::Star}, {"/", OperatorKind::Slash},
        {"%", OperatorKind::Percent}, {"&&", OperatorKind::LogicalAnd}, {"||", OperatorKind::LogicalOr},
        {"!", OperatorKind::LogicalNot}, {"&", OperatorKind::BitAnd}, {"|", OperatorKind::BitOr},
        {"^", OperatorKind::BitXor}, {"~", OperatorKind::BitNot}, {"<<", OperatorKind::ShiftLeft},
        {">>", OperatorKind::ShiftRight}, {"=", OperatorKind::Assign}, {"+=", OperatorKind::PlusAssign},
        {"-=", OperatorKind::MinusAssign}, {"*=", OperatorKind::MulAssign}, {"/=", OperatorKind::DivAssign},
        {"%=", OperatorKind::ModAssign}, {"&=", OperatorKind::AndAssign}, {"|=", OperatorKind::OrAssign},
        {"^=", OperatorKind::XorAssign}, {"<<=", OperatorKind::ShlAssign}, {">>=", OperatorKind::ShrAssign},
        {"==", OperatorKind::Equal}, {"!=", OperatorKind::NotEqual}, {"<", OperatorKind::Less},
        {"<=", OperatorKind::LessEqual}, {">", OperatorKind::Greater}, {">=", OperatorKind::GreaterEqual},
        {"++", OperatorKind::Increment}, {"--", OperatorKind::Decrement}, {"->", OperatorKind::Arrow},
        {".", OperatorKind::Dot}, {"::", OperatorKind::Scope}, {".*", OperatorKind::MemberPtr},
        {"->*", OperatorKind::MemberPtr}, {"sizeof", OperatorKind::SizeofOp}, {"alignof", OperatorKind::AlignofOp},
        {"typeid", OperatorKind::TypeidOp}
    };
    return tbl;
}

static const std::vector<std::pair<std::string, PunctuationKind>>& punctuation_table() {
    static std::vector<std::pair<std::string, PunctuationKind>> tbl = {
        {"(", PunctuationKind::LParen}, {")", PunctuationKind::RParen},
        {"[", PunctuationKind::LBracket}, {"]", PunctuationKind::RBracket},
        {"{", PunctuationKind::LBrace}, {"}", PunctuationKind::RBrace},
        {";", PunctuationKind::Semicolon}, {":", PunctuationKind::Colon},
        {",", PunctuationKind::Comma}, {"...", PunctuationKind::Ellipsis},
        {"?", PunctuationKind::Question}, {"<", PunctuationKind::TemplateLT}, {">", PunctuationKind::TemplateGT}
    };
    return tbl;
}

Token make_primitive_token(PrimitiveKind kind,
                           bool is_signed,
                           bool is_const,
                           bool is_volatile,
                           std::uint32_t array_size,
                           const SourceLocation& loc,
                           const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Primitive;
    t.location = loc;
    t.lexeme = lexeme;
    // primitive is POD-like; assigning is fine
    t.primitive.kind = kind;
    t.primitive.is_signed = is_signed;
    t.primitive.is_const = is_const;
    t.primitive.is_volatile = is_volatile;
    t.primitive.array_size = array_size;
    return t;
}

Token make_container_token(ContainerKind kind,
                           const SourceLocation& loc,
                           const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Container;
    t.location = loc;
    t.lexeme = lexeme;
    t.container.kind = kind;
    return t;
}

Token make_keyword_token(KeywordKind kind,
                         const SourceLocation& loc,
                         const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Keyword;
    t.location = loc;
    t.lexeme = lexeme;
    t.keyword = kind;
    return t;
}

Token make_operator_token(OperatorKind kind,
                          const SourceLocation& loc,
                          const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Operator;
    t.location = loc;
    t.lexeme = lexeme;
    t.op = kind;
    return t;
}

Token make_punctuation_token(PunctuationKind kind,
                             const SourceLocation& loc,
                             const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Punctuation;
    t.location = loc;
    t.lexeme = lexeme;
    t.punct = kind;
    return t;
}

Token make_literal_int(std::int64_t value,
                       const SourceLocation& loc,
                       const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Literal;
    t.location = loc;
    t.lexeme = lexeme;
    // construct literal in union
    new (&t.literal) LiteralData();
    t.literal.kind = LiteralKind::Integer;
    t.literal.int_val = value;
    return t;
}

Token make_literal_float(double value,
                         const SourceLocation& loc,
                         const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Literal;
    t.location = loc;
    t.lexeme = lexeme;
    new (&t.literal) LiteralData();
    t.literal.kind = LiteralKind::Float;
    t.literal.float_val = value;
    return t;
}

Token make_literal_char(char value,
                        const SourceLocation& loc,
                        const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Literal;
    t.location = loc;
    t.lexeme = lexeme;
    new (&t.literal) LiteralData();
    t.literal.kind = LiteralKind::Char;
    t.literal.char_val = value;
    return t;
}

Token make_literal_string(const std::string& value,
                          const SourceLocation& loc,
                          const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Literal;
    t.location = loc;
    t.lexeme = lexeme;
    new (&t.literal) LiteralData();
    t.literal.kind = LiteralKind::String;
    // string_val is non-trivial — assign
    t.literal.string_val = value;
    return t;
}

Token make_literal_bool(bool value,
                        const SourceLocation& loc,
                        const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Literal;
    t.location = loc;
    t.lexeme = lexeme;
    new (&t.literal) LiteralData();
    t.literal.kind = LiteralKind::Bool;
    t.literal.int_val = value ? 1 : 0;
    return t;
}

Token make_literal_null(const SourceLocation& loc,
                        const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Literal;
    t.location = loc;
    t.lexeme = lexeme;
    new (&t.literal) LiteralData();
    t.literal.kind = LiteralKind::Null;
    t.literal.int_val = 0;
    return t;
}

Token make_identifier_token(const SourceLocation& loc,
                            const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Identifier;
    t.location = loc;
    t.lexeme = lexeme;
    return t;
}

Token make_preprocessor_token(PreprocessorKind kind,
                              const SourceLocation& loc,
                              const std::string& lexeme) {
    Token t;
    t.category = TokenCategory::Preprocessor;
    t.location = loc;
    t.lexeme = lexeme;
    t.pp = kind;
    return t;
}

void destroy_token(Token& t) {
    // Destroy non-trivial union member if necessary
    if (t.category == TokenCategory::Literal) {
        // Call destructor of LiteralData explicitly (it contains std::string)
        reinterpret_cast<LiteralData*>(&t.literal)->~LiteralData();
    }
    // Clear lexeme string (outside union)
    t.lexeme.clear();
    // Mark token as Unknown to avoid double-destroy attempts
    t.category = TokenCategory::Unknown;
}

Token copy_token(const Token& src) {
    Token dst;
    dst.category = src.category;
    dst.location = src.location;
    dst.lexeme = src.lexeme;

    switch (src.category) {
        case TokenCategory::Primitive:
            dst.primitive = src.primitive; // POD copy
            break;
        case TokenCategory::Container:
            dst.container = src.container;
            break;
        case TokenCategory::Keyword:
            dst.keyword = src.keyword;
            break;
        case TokenCategory::Operator:
            dst.op = src.op;
            break;
        case TokenCategory::Punctuation:
            dst.punct = src.punct;
            break;
        case TokenCategory::Preprocessor:
            dst.pp = src.pp;
            break;
        case TokenCategory::Literal:
            // placement-new then copy fields manually
            new (&dst.literal) LiteralData();
            dst.literal.kind = src.literal.kind;
            switch (src.literal.kind) {
                case LiteralKind::Integer:
                case LiteralKind::Bool:
                case LiteralKind::Null:
                    dst.literal.int_val = src.literal.int_val;
                    break;
                case LiteralKind::Float:
                    dst.literal.float_val = src.literal.float_val;
                    break;
                case LiteralKind::Char:
                    dst.literal.char_val = src.literal.char_val;
                    break;
                case LiteralKind::String:
                    dst.literal.string_val = src.literal.string_val;
                    break;
            }
            break;
        case TokenCategory::Identifier:
            // nothing in union to copy
            break;
        case TokenCategory::EndOfFile:
        case TokenCategory::Unknown:
        default:
            break;
    }
    return dst;
}

Token move_token(Token&& src) {
    Token dst;
    dst.category = src.category;
    dst.location = src.location;
    dst.lexeme = std::move(src.lexeme);

    switch (src.category) {
        case TokenCategory::Primitive:
            dst.primitive = src.primitive;
            break;
        case TokenCategory::Container:
            dst.container = src.container;
            break;
        case TokenCategory::Keyword:
            dst.keyword = src.keyword;
            break;
        case TokenCategory::Operator:
            dst.op = src.op;
            break;
        case TokenCategory::Punctuation:
            dst.punct = src.punct;
            break;
        case TokenCategory::Preprocessor:
            dst.pp = src.pp;
            break;
        case TokenCategory::Literal:
            new (&dst.literal) LiteralData();
            dst.literal.kind = src.literal.kind;
            switch (src.literal.kind) {
                case LiteralKind::Integer:
                case LiteralKind::Bool:
                case LiteralKind::Null:
                    dst.literal.int_val = src.literal.int_val;
                    break;
                case LiteralKind::Float:
                    dst.literal.float_val = src.literal.float_val;
                    break;
                case LiteralKind::Char:
                    dst.literal.char_val = src.literal.char_val;
                    break;
                case LiteralKind::String:
                    dst.literal.string_val = std::move(const_cast<LiteralData&>(src.literal).string_val);
                    break;
            }
            // destroy the source literal's string so we don't double-free on its destroy_token
            reinterpret_cast<LiteralData*>(&src.literal)->~LiteralData();
            break;
        case TokenCategory::Identifier:
            // nothing in union to move
            break;
        case TokenCategory::EndOfFile:
        case TokenCategory::Unknown:
        default:
            break;
    }

    // leave src in safe state
    src.lexeme.clear();
    src.category = TokenCategory::Unknown;
    return dst;
}

const char* primitive_to_string(PrimitiveKind kind) {
    switch (kind) {
        case PrimitiveKind::Void:   return "void";
        case PrimitiveKind::Char:   return "char";
        case PrimitiveKind::Short:  return "short";
        case PrimitiveKind::Int:    return "int";
        case PrimitiveKind::Long:   return "long";
        case PrimitiveKind::Float:  return "float";
        case PrimitiveKind::Double: return "double";
        case PrimitiveKind::Bool:   return "bool";
        case PrimitiveKind::Pointer:return "pointer";
        case PrimitiveKind::Array:  return "array";
        default: return "unknown_primitive";
    }
}

const char* container_to_string(ContainerKind kind) {
    switch (kind) {
        case ContainerKind::String:     return "string";
        case ContainerKind::Vector:     return "vector";
        case ContainerKind::LinkedList: return "list";
        case ContainerKind::Stack:      return "stack";
        case ContainerKind::Queue:      return "queue";
        default: return "unknown_container";
    }
}

const char* keyword_to_string(KeywordKind kind) {
    // Only a subset mapped for readability; extend as needed
    switch (kind) {
        case KeywordKind::Auto: return "auto";
        case KeywordKind::Break: return "break";
        case KeywordKind::Case: return "case";
        case KeywordKind::Const: return "const";
        case KeywordKind::Continue: return "continue";
        case KeywordKind::Default: return "default";
        case KeywordKind::Do: return "do";
        case KeywordKind::Else: return "else";
        case KeywordKind::Enum: return "enum";
        case KeywordKind::Extern: return "extern";
        case KeywordKind::For: return "for";
        case KeywordKind::Goto: return "goto";
        case KeywordKind::If: return "if";
        case KeywordKind::Inline: return "inline";
        case KeywordKind::Register: return "register";
        case KeywordKind::Restrict: return "restrict";
        case KeywordKind::Return: return "return";
        case KeywordKind::Signed: return "signed";
        case KeywordKind::Sizeof: return "sizeof";
        case KeywordKind::Static: return "static";
        case KeywordKind::Struct: return "struct";
        case KeywordKind::Switch: return "switch";
        case KeywordKind::Typedef: return "typedef";
        case KeywordKind::Union: return "union";
        case KeywordKind::Unsigned: return "unsigned";
        case KeywordKind::Volatile: return "volatile";
        case KeywordKind::While: return "while";
        case KeywordKind::VoidKw: return "void";
        case KeywordKind::IntKw: return "int";
        case KeywordKind::FloatKw: return "float";
        case KeywordKind::DoubleKw: return "double";
        case KeywordKind::CharKw: return "char";
        case KeywordKind::LongKw: return "long";
        case KeywordKind::ShortKw: return "short";

        // C++ subset:
        case KeywordKind::Class: return "class";
        case KeywordKind::Namespace: return "namespace";
        case KeywordKind::New: return "new";
        case KeywordKind::Nullptr: return "nullptr";
        case KeywordKind::Operator: return "operator";
        case KeywordKind::Template: return "template";
        case KeywordKind::This: return "this";
        case KeywordKind::Using: return "using";
        case KeywordKind::Virtual: return "virtual";

        default: return "keyword";
    }
}

const char* operator_to_string(OperatorKind k) {
    switch (k) {
        case OperatorKind::Plus: return "+";
        case OperatorKind::Minus: return "-";
        case OperatorKind::Star: return "*";
        case OperatorKind::Slash: return "/";
        case OperatorKind::Percent: return "%";
        case OperatorKind::LogicalAnd: return "&&";
        case OperatorKind::LogicalOr: return "||";
        case OperatorKind::LogicalNot: return "!";
        case OperatorKind::BitAnd: return "&";
        case OperatorKind::BitOr: return "|";
        case OperatorKind::BitXor: return "^";
        case OperatorKind::BitNot: return "~";
        case OperatorKind::ShiftLeft: return "<<";
        case OperatorKind::ShiftRight: return ">>";
        case OperatorKind::Assign: return "=";
        case OperatorKind::PlusAssign: return "+=";
        case OperatorKind::MinusAssign: return "-=";
        case OperatorKind::MulAssign: return "*=";
        case OperatorKind::DivAssign: return "/=";
        case OperatorKind::ModAssign: return "%=";
        case OperatorKind::AndAssign: return "&=";
        case OperatorKind::OrAssign: return "|=";
        case OperatorKind::XorAssign: return "^=";
        case OperatorKind::ShlAssign: return "<<=";
        case OperatorKind::ShrAssign: return ">>=";
        case OperatorKind::Equal: return "==";
        case OperatorKind::NotEqual: return "!=";
        case OperatorKind::Less: return "<";
        case OperatorKind::LessEqual: return "<=";
        case OperatorKind::Greater: return ">";
        case OperatorKind::GreaterEqual: return ">=";
        case OperatorKind::Increment: return "++";
        case OperatorKind::Decrement: return "--";
        case OperatorKind::Arrow: return "->";
        case OperatorKind::Dot: return ".";
        case OperatorKind::Scope: return "::";
        case OperatorKind::MemberPtr: return ".* / ->*";
        case OperatorKind::SizeofOp: return "sizeof";
        case OperatorKind::AlignofOp: return "alignof";
        case OperatorKind::TypeidOp: return "typeid";
        default: return "op";
    }
}

const char* punctuation_to_string(PunctuationKind p) {
    switch (p) {
        case PunctuationKind::LParen: return "(";
        case PunctuationKind::RParen: return ")";
        case PunctuationKind::LBracket: return "[";
        case PunctuationKind::RBracket: return "]";
        case PunctuationKind::LBrace: return "{";
        case PunctuationKind::RBrace: return "}";
        case PunctuationKind::Semicolon: return ";";
        case PunctuationKind::Colon: return ":";
        case PunctuationKind::Comma: return ",";
        case PunctuationKind::Ellipsis: return "...";
        case PunctuationKind::Question: return "?";
        case PunctuationKind::TemplateLT: return "<";
        case PunctuationKind::TemplateGT: return ">";
        default: return "punct";
    }
}

const char* literal_to_string(LiteralKind k) {
    switch (k) {
        case LiteralKind::Integer: return "integer";
        case LiteralKind::Float: return "float";
        case LiteralKind::Char: return "char";
        case LiteralKind::String: return "string";
        case LiteralKind::Bool: return "bool";
        case LiteralKind::Null: return "null";
        default: return "literal";
    }
}

const char* preprocessor_to_string(PreprocessorKind k) {
    switch (k) {
        case PreprocessorKind::Define: return "#define";
        case PreprocessorKind::Undef: return "#undef";
        case PreprocessorKind::If: return "#if";
        case PreprocessorKind::Ifdef: return "#ifdef";
        case PreprocessorKind::Ifndef: return "#ifndef";
        case PreprocessorKind::Elif: return "#elif";
        case PreprocessorKind::Else: return "#else";
        case PreprocessorKind::Endif: return "#endif";
        case PreprocessorKind::Line: return "#line";
        case PreprocessorKind::Error: return "#error";
        case PreprocessorKind::Pragma: return "#pragma";
        case PreprocessorKind::Include: return "#include";
        default: return "#pp";
    }
}

const char* token_category_to_string(TokenCategory cat) {
    switch (cat) {
        case TokenCategory::Primitive: return "Primitive";
        case TokenCategory::Container: return "Container";
        case TokenCategory::Keyword: return "Keyword";
        case TokenCategory::Operator: return "Operator";
        case TokenCategory::Punctuation: return "Punctuation";
        case TokenCategory::Literal: return "Literal";
        case TokenCategory::Identifier: return "Identifier";
        case TokenCategory::Preprocessor: return "Preprocessor";
        case TokenCategory::EndOfFile: return "EOF";
        case TokenCategory::Unknown: return "Unknown";
        default: return "Cat";
    }
}

std::string token_to_string(const Token& t) {
    std::ostringstream oss;
    oss << token_category_to_string(t.category) << " ";

    // location if available
    oss << "(" << t.location.line << ":" << t.location.column << ") ";

    // lexeme
    if (!t.lexeme.empty()) {
        oss << "'" << t.lexeme << "' ";
    }

    switch (t.category) {
        case TokenCategory::Primitive:
            oss << primitive_to_string(t.primitive.kind);
            break;
        case TokenCategory::Container:
            oss << container_to_string(t.container.kind);
            break;
        case TokenCategory::Keyword:
            oss << keyword_to_string(t.keyword);
            break;
        case TokenCategory::Operator:
            oss << operator_to_string(t.op);
            break;
        case TokenCategory::Punctuation:
            oss << punctuation_to_string(t.punct);
            break;
        case TokenCategory::Preprocessor:
            oss << preprocessor_to_string(t.pp);
            break;
        case TokenCategory::Literal:
            oss << literal_to_string(t.literal.kind) << " ";
            switch (t.literal.kind) {
                case LiteralKind::Integer:
                    oss << t.literal.int_val;
                    break;
                case LiteralKind::Float:
                    oss << t.literal.float_val;
                    break;
                case LiteralKind::Char:
                    oss << "'" << t.literal.char_val << "'";
                    break;
                case LiteralKind::String:
                    oss << "\"" << t.literal.string_val << "\"";
                    break;
                case LiteralKind::Bool:
                    oss << (t.literal.int_val ? "true" : "false");
                    break;
                case LiteralKind::Null:
                    oss << "null";
                    break;
            }
            break;
        case TokenCategory::Identifier:
            // already printed lexeme
            break;
        case TokenCategory::EndOfFile:
            oss << "<eof>";
            break;
        case TokenCategory::Unknown:
        default:
            oss << "<unknown>";
            break;
    }

    return oss.str();
}

void print_token(const Token& t) {
    std::cout << token_to_string(t) << std::endl;
}

// ---------------------------------------------------------------------------
// Lookups: string -> enum
// (linear lookup for readability; replace with hash map if performance needed)
// ---------------------------------------------------------------------------
bool lookup_keyword(const std::string& text, KeywordKind& out_kind) {
    for (auto &p : keyword_table()) {
        if (p.first == text) {
            out_kind = p.second;
            return true;
        }
    }
    return false;
}

bool lookup_operator(const std::string& text, OperatorKind& out_kind) {
    for (auto &p : operator_table()) {
        if (p.first == text) {
            out_kind = p.second;
            return true;
        }
    }
    return false;
}

bool lookup_punctuation(const std::string& text, PunctuationKind& out_kind) {
    for (auto &p : punctuation_table()) {
        if (p.first == text) {
            out_kind = p.second;
            return true;
        }
    }
    return false;
}

bool is_primitive(const Token& t) {
    return t.category == TokenCategory::Primitive;
}

bool is_container(const Token& t) {
    return t.category == TokenCategory::Container;
}

bool is_keyword(const Token& t) {
    return t.category == TokenCategory::Keyword;
}

bool is_operator(const Token& t) {
    return t.category == TokenCategory::Operator;
}

bool is_punctuation(const Token& t) {
    return t.category == TokenCategory::Punctuation;
}

bool is_literal(const Token& t) {
    return t.category == TokenCategory::Literal;
}

bool is_identifier(const Token& t) {
    return t.category == TokenCategory::Identifier;
}

bool is_preprocessor(const Token& t) {
    return t.category == TokenCategory::Preprocessor;
}

// A small set of keywords that represent types (C + C++)
bool is_type_keyword(const Token& t) {
    if (t.category != TokenCategory::Keyword) return false;
    switch (t.keyword) {
        case KeywordKind::VoidKw:
        case KeywordKind::CharKw:
        case KeywordKind::IntKw:
        case KeywordKind::FloatKw:
        case KeywordKind::DoubleKw:
        case KeywordKind::LongKw:
        case KeywordKind::ShortKw:
        case KeywordKind::BoolKw:
        case KeywordKind::Struct:
        case KeywordKind::Union:
        case KeywordKind::Enum:
            return true;
        default:
            return false;
    }
}

bool is_unary_operator(const Token& t) {
    if (t.category != TokenCategory::Operator) return false;
    switch (t.op) {
        case OperatorKind::Plus:
        case OperatorKind::Minus:
        case OperatorKind::LogicalNot:
        case OperatorKind::BitNot:
        case OperatorKind::Increment:
        case OperatorKind::Decrement:
        case OperatorKind::Star: // when used as deref
        case OperatorKind::BitAnd: // when used as address-of
        case OperatorKind::SizeofOp:
        case OperatorKind::AlignofOp:
        case OperatorKind::TypeidOp:
            return true;
        default:
            return false;
    }
}

bool is_binary_operator(const Token& t) {
    if (t.category != TokenCategory::Operator) return false;
    switch (t.op) {
        case OperatorKind::Plus:
        case OperatorKind::Minus:
        case OperatorKind::Star:
        case OperatorKind::Slash:
        case OperatorKind::Percent:
        case OperatorKind::LogicalAnd:
        case OperatorKind::LogicalOr:
        case OperatorKind::BitAnd:
        case OperatorKind::BitOr:
        case OperatorKind::BitXor:
        case OperatorKind::ShiftLeft:
        case OperatorKind::ShiftRight:
        case OperatorKind::Assign:
        case OperatorKind::PlusAssign:
        case OperatorKind::MinusAssign:
        case OperatorKind::MulAssign:
        case OperatorKind::DivAssign:
        case OperatorKind::ModAssign:
        case OperatorKind::AndAssign:
        case OperatorKind::OrAssign:
        case OperatorKind::XorAssign:
        case OperatorKind::ShlAssign:
        case OperatorKind::ShrAssign:
        case OperatorKind::Equal:
        case OperatorKind::NotEqual:
        case OperatorKind::Less:
        case OperatorKind::LessEqual:
        case OperatorKind::Greater:
        case OperatorKind::GreaterEqual:
        case OperatorKind::Arrow:
        case OperatorKind::Dot:
        case OperatorKind::MemberPtr:
            return true;
        default:
            return false;
    }
}

} // namespace token
