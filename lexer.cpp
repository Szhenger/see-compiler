#include "token.hpp"
#include "lexer.hpp"

#include <cctype>
#include <sstream>
#include <stdexcept>

Lexer::Lexer(const std::string& input)
    : source(input), index(0), line(1), column(1)
{}

char Lexer::current_char() const {
    if (index >= source.size()) return '\0';
    return source[index];
}

char Lexer::peek_char(int ahead) const {
    size_t pos = index + (ahead > 0 ? static_cast<size_t>(ahead) : 0);
    if (pos >= source.size()) return '\0';
    return source[pos];
}

void Lexer::advance(int count) {
    for (int i = 0; i < count; ++i) {
        if (index >= source.size()) return;
        char c = source[index++];
        if (c == '\n') {
            ++line;
            column = 1;
        } else {
            ++column;
        }
    }
}

bool Lexer::is_eof() const {
    return index >= source.size();
}

void Lexer::skip_whitespace() {
    while (!is_eof()) {
        char c = current_char();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance(1);
            continue;
        }
        if (c == '/') {
            char p = peek_char(1);
            if (p == '/' || p == '*') {
                skip_comment();
                continue;
            }
        }
        break;
    }
}

void Lexer::skip_comment() {
    if (current_char() != '/') return;
    char next = peek_char(1);
    // single-line comment
    if (next == '/') {
        // advance "//"
        advance(2);
        while (!is_eof() && current_char() != '\n') advance(1);
        // consume newline if present
        if (!is_eof() && current_char() == '\n') advance(1);
        return;
    }
    // block comment "/* ... */"
    if (next == '*') {
        advance(2);
        while (!is_eof()) {
            if (current_char() == '*' && peek_char(1) == '/') {
                // consume "*/"
                advance(2);
                return;
            } else {
                advance(1);
            }
        }
        // Unterminated comment: we just return (lexer user can emit diagnostic later)
        return;
    }
}

static Token make_eof_token(const SourceLocation& loc) {
    Token t;
    t.category = TokenCategory::EndOfFile;
    t.location = loc;
    t.lexeme = "";
    return t;
}

Token Lexer::lex_identifier_or_keyword() {
    size_t start = index;
    int start_line = line;
    int start_col = column;

    // identifiers: [A-Za-z_][A-Za-z0-9_]*
    while (!is_eof()) {
        char c = current_char();
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            advance(1);
        } else {
            break;
        }
    }
    std::string text = source.substr(start, index - start);

    SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                        static_cast<std::uint32_t>(start_col) };

    KeywordKind kk;
    if (lookup_keyword(text, kk)) {
        return make_keyword_token(kk, loc, text);
    }

    // not a keyword -> identifier
    return make_identifier_token(loc, text);
}

Token Lexer::lex_number_literal() {
    size_t start = index;
    int start_line = line;
    int start_col = column;

    bool seen_dot = false;
    bool seen_exp = false;

    // integer or floating: simple DFA
    while (!is_eof()) {
        char c = current_char();
        if (std::isdigit(static_cast<unsigned char>(c))) {
            advance(1);
            continue;
        }
        if (c == '.' && !seen_dot && !seen_exp) {
            seen_dot = true;
            advance(1);
            continue;
        }
        if ((c == 'e' || c == 'E') && !seen_exp) {
            seen_exp = true;
            advance(1);
            if (current_char() == '+' || current_char() == '-') advance(1);
            continue;
        }
        break;
    }

    std::string text = source.substr(start, index - start);
    SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                        static_cast<std::uint32_t>(start_col) };

    // decide int vs float
    if (seen_dot || seen_exp) {
        // parse double
        try {
            double v = std::stod(text);
            return make_literal_float(v, loc, text);
        } catch (...) {
            // fallback: return as string literal token of float kind with 0 value
            return make_literal_float(0.0, loc, text);
        }
    } else {
        // integer
        try {
            std::int64_t iv = std::stoll(text, nullptr, 0); // auto base if 0x, 0b, 0 prefix
            return make_literal_int(iv, loc, text);
        } catch (...) {
            return make_literal_int(0, loc, text);
        }
    }
}

static char parse_escape_sequence(Lexer& L) {
    // assumes current char is the character after backslash (we've advanced to it)
    char c = L.current_char();
    char out = c;
    switch (c) {
        case '\\': out = '\\'; break;
        case 'n': out = '\n'; break;
        case 't': out = '\t'; break;
        case 'r': out = '\r'; break;
        case '\'': out = '\''; break;
        case '\"': out = '\"'; break;
        case '0': out = '\0'; break;
        default: out = c; break; // unknown escape -> literal
    }
    // consume the escape char
    L.advance(1);
    return out;
}

Token Lexer::lex_string_literal() {
    // handle "..." and raw forms could be added later
    size_t start = index;
    int start_line = line;
    int start_col = column;

    // consume opening quote
    advance(1);

    std::string value;
    while (!is_eof()) {
        char c = current_char();
        if (c == '"') {
            advance(1); // consume closing quote
            break;
        }
        if (c == '\\') {
            advance(1); // consume backslash
            if (is_eof()) break;
            char esc = parse_escape_sequence(*this);
            value.push_back(esc);
            continue;
        }
        value.push_back(c);
        advance(1);
    }

    std::string lexeme = source.substr(start, index - start);
    SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                        static_cast<std::uint32_t>(start_col) };

    return make_literal_string(value, loc, lexeme);
}

Token Lexer::lex_char_literal() {
    // 'a' or '\n' or '\xNN' simple handling
    size_t start = index;
    int start_line = line;
    int start_col = column;

    advance(1); // consume opening '

    char value = '\0';
    if (!is_eof()) {
        char c = current_char();
        if (c == '\\') {
            advance(1);
            value = parse_escape_sequence(*this);
        } else {
            value = c;
            advance(1);
        }
    }

    // consume closing '
    if (current_char() == '\'') {
        advance(1);
    }

    std::string lexeme = source.substr(start, index - start);
    SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                        static_cast<std::uint32_t>(start_col) };

    return make_literal_char(value, loc, lexeme);
}

Token Lexer::lex_operator() {
    size_t start = index;
    int start_line = line;
    int start_col = column;

    // maximum operator length we'll try
    for (int len = 3; len >= 1; --len) {
        if (index + len <= source.size()) {
            std::string s = source.substr(index, len);
            OperatorKind ok;
            if (lookup_operator(s, ok)) {
                // consume len chars
                advance(len);
                SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                                    static_cast<std::uint32_t>(start_col) };
                return make_operator_token(ok, loc, s);
            }
        }
    }

    // fallback: single-char operator (treat as punctuation)
    char c = current_char();
    std::string s(1, c);
    advance(1);
    SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                        static_cast<std::uint32_t>(start_col) };

    PunctuationKind pk;
    if (lookup_punctuation(s, pk)) {
        return make_punctuation_token(pk, loc, s);
    }

    // Unknown single char -> return as identifier-like token
    return make_identifier_token(loc, s);
}

Token Lexer::lex_punctuation() {
    size_t start = index;
    int start_line = line;
    int start_col = column;

    // try length 3 punctuation (ellipsis)
    if (index + 3 <= source.size()) {
        std::string s = source.substr(index, 3);
        PunctuationKind pk;
        if (lookup_punctuation(s, pk)) {
            advance(3);
            SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                                static_cast<std::uint32_t>(start_col) };
            return make_punctuation_token(pk, loc, s);
        }
    }
    // try length 2
    if (index + 2 <= source.size()) {
        std::string s = source.substr(index, 2);
        PunctuationKind pk;
        if (lookup_punctuation(s, pk)) {
            advance(2);
            SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                                static_cast<std::uint32_t>(start_col) };
            return make_punctuation_token(pk, loc, s);
        }
    }
    // try length 1
    if (index + 1 <= source.size()) {
        std::string s = source.substr(index, 1);
        PunctuationKind pk;
        if (lookup_punctuation(s, pk)) {
            advance(1);
            SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                                static_cast<std::uint32_t>(start_col) };
            return make_punctuation_token(pk, loc, s);
        }
    }

    // fallback: unknown punctuation as identifier token
    char c = current_char();
    std::string s(1, c);
    advance(1);
    SourceLocation loc{ static_cast<std::uint32_t>(start_line),
                        static_cast<std::uint32_t>(start_col) };
    return make_identifier_token(loc, s);
}

Token Lexer::peek_token() {
    Lexer copy = *this; // uses memberwise copy; safe for current design
    return copy.next_token();
}

std::vector<Token> Lexer::tokenize_all() {
    std::vector<Token> out;
    while (!is_eof()) {
        Token t = next_token();
        out.push_back(t);
        if (t.category == TokenCategory::EndOfFile) break;
    }
    return out;
}

Token Lexer::next_token() {
    skip_whitespace();

    // record start location for this token
    SourceLocation loc{ static_cast<std::uint32_t>(line),
                        static_cast<std::uint32_t>(column) };

    if (is_eof()) {
        return make_eof_token(loc);
    }

    char c = current_char();

    // preprocessor directive at start of line (simple heuristic)
    if (c == '#' && (column == 1)) {
        // consume until newline
        size_t start = index;
        while (!is_eof() && current_char() != '\n') advance(1);
        std::string lex = source.substr(start, index - start);
        return make_preprocessor_token(PreprocessorKind::Include, loc, lex); // use Include generically
    }

    // identifiers and keywords
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        return lex_identifier_or_keyword();
    }

    // numbers
    if (std::isdigit(static_cast<unsigned char>(c))) {
        return lex_number_literal();
    }

    // string literal
    if (c == '"') {
        return lex_string_literal();
    }

    // char literal
    if (c == '\'') {
        return lex_char_literal();
    }

    // operators and punctuation
    // many single characters overlap; decide by char sets
    // quick classification: symbols that could be operators
    const char operator_start_chars[] = "+-*/%&|^~!=<>.:";
    bool could_be_op = false;
    for (char oc : operator_start_chars) {
        if (c == oc) { could_be_op = true; break; }
    }

    if (could_be_op) {
        return lex_operator();
    }

    // punctuation: parentheses, braces, comma, semicolon, question, colon, etc.
    const char punct_chars[] = "();,:{}[]?<>"; // note '<' and '>' might be template tokens
    bool could_be_punct = false;
    for (char pc : punct_chars) {
        if (c == pc) { could_be_punct = true; break; }
    }

    if (could_be_punct) {
        return lex_punctuation();
    }

    // whitespace/unknown single char fallback: create identifier-like token
    // consume one char and return
    std::string s(1, c);
    advance(1);
    return make_identifier_token(loc, s);
}
