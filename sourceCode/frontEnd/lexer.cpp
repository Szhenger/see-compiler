#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cctype>
#include <string_view>
#include "token.hpp"

namespace see {

// ----------------------------- Token Kinds -----------------------------

enum class TokKind : uint16_t {
  End,            // EOF
  Identifier,
  Keyword,        // non-primitive keywords (e.g., if/else/return/typedef/struct/union/enum/for/while/auto/extern/inline)
  Type,           // canonical primitive type (via token.cpp Prim)
  Integer,
  Floating,
  CharLit,
  StringLit,
  Preprocessor,   // line that begins with '#'
  Operator,       // operators & punctuators, incl. separators
  Unknown
};

struct Token {
  TokKind           kind;
  std::string_view  lexeme;     // view into original buffer
  uint32_t          line;       // 1-based
  uint32_t          column;     // 1-based
  // Optional payload for TYPE tokens
  Prim              prim;       // valid iff kind==Type
};

// ----------------------------- Lexer Core ------------------------------

class Lexer {
public:
  Lexer(const char* src, size_t len)
  : begin_(src), cur_(src), end_(src + len), line_(1), col_(1) {}

  // Return next token (EOF returns TokKind::End)
  Token next() {
    skip_ws_and_comments();

    const uint32_t tok_line = line_;
    const uint32_t tok_col  = col_;
    const char* tok_start   = cur_;

    if (cur_ >= end_) return make_simple(TokKind::End, tok_start, 0, tok_line, tok_col);

    // Preprocessor line (only if '#' is the first non-space on a line)
    if (*cur_ == '#' && at_line_start_) {
      const char* start = cur_;
      while (cur_ < end_ && *cur_ != '\n') advance();
      return make_simple(TokKind::Preprocessor, start, size_t(cur_ - start), tok_line, tok_col);
    }

    // Identifiers / keywords / primitive-type collapsing
    if (is_ident_start(*cur_)) {
      Token t = scan_identifier_or_type();
      t.line = tok_line; t.column = tok_col;
      return t;
    }

    // Numeric literals
    if (std::isdigit(unsigned(*cur_)) || (cur_+1 < end_ && *cur_ == '.' && std::isdigit(unsigned(*(cur_+1))))) {
      Token t = scan_number();
      t.line = tok_line; t.column = tok_col;
      return t;
    }

    // String / char literals (with optional u8/u/U/L prefixes)
    if (is_string_or_char_prefix()) {
      Token t = scan_string_or_char();
      t.line = tok_line; t.column = tok_col;
      return t;
    }

    // Operators & punctuators
    Token op = scan_operator_or_punct();
    op.line = tok_line; op.column = tok_col;
    if (op.kind != TokKind::Unknown) return op;

    // Fallback: unknown single char
    const char* s = cur_;
    advance();
    return make_simple(TokKind::Unknown, s, 1, tok_line, tok_col);
  }

private:
  // ------------- character stream helpers -------------
  inline bool eof() const { return cur_ >= end_; }
  inline char peek() const { return eof() ? '\0' : *cur_; }
  inline char peek_next() const { return (cur_+1 < end_) ? *(cur_+1) : '\0'; }

  inline void advance() {
    if (cur_ >= end_) return;
    char c = *cur_++;
    if (c == '\n') { line_++; col_ = 1; at_line_start_ = true; }
    else { col_++; at_line_start_ = false; }
  }

  inline bool match(char c) {
    if (cur_ < end_ && *cur_ == c) { advance(); return true; }
    return false;
  }

  static inline bool is_ident_start(char c) {
    return std::isalpha(unsigned(c)) || c == '_' ;
  }
  static inline bool is_ident_char(char c) {
    return std::isalnum(unsigned(c)) || c == '_' ;
  }

  // ------------- token makers -------------

  static inline Token make_simple(TokKind k, const char* s, size_t n, uint32_t ln, uint32_t col) {
    Token t;
    t.kind = k;
    t.lexeme = std::string_view{s, n};
    t.line = ln; t.column = col;
    t.prim = Prim::Void; // default; ignore unless Type
    return t;
  }

  // ------------- skipping -------------

  void skip_ws_and_comments() {
    bool again = true;
    while (again && cur_ < end_) {
      again = false;

      // whitespace
      while (cur_ < end_) {
        char c = *cur_;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') { advance(); }
        else if (c == '\n') { advance(); /* at_line_start_ handled in advance() */ }
        else break;
      }

      // comments
      if (cur_+1 < end_ && *cur_ == '/' && *(cur_+1) == '/') {
        // line comment
        advance(); advance();
        while (cur_ < end_ && *cur_ != '\n') advance();
        again = true;
      } else if (cur_+1 < end_ && *cur_ == '/' && *(cur_+1) == '*') {
        // block comment
        advance(); advance();
        while (cur_ < end_) {
          if (cur_+1 < end_ && *cur_ == '*' && *(cur_+1) == '/') { advance(); advance(); break; }
          advance();
        }
        again = true;
      }
    }
  }

  // ------------- identifiers / keywords / types -------------

  // Simple C/C++ keyword set (non-primitive). Primitive words are handled via token.cpp.
  static bool is_keyword(std::string_view s) {
    static const char *kw[] = {
      "if","else","switch","case","default","break","continue","return",
      "for","while","do","goto",
      "typedef","struct","union","enum","sizeof","alignof",
      "auto","register","static","extern","const","volatile","restrict",
      "inline","_Noreturn","_Alignas","_Alignof","_Atomic","_Thread_local",
      "namespace","using","class","template","typename","new","delete",
      "try","catch","throw","constexpr","consteval","constinit","explicit",
      "friend","operator","private","protected","public","virtual","override",
      "mutable","noexcept","static_assert"
    };
    for (const char *k : kw) {
      if (s == k) return true;
    }
    return false;
  }

  bool try_scan_primitive_collapse(Prim *out_prim, const char **out_start, const char **out_end) {
    const char *save = cur_;
    uint32_t save_line = line_, save_col = col_;
    bool save_at_bol = at_line_start_;

    char probe[128]; size_t p = 0;
    const char *first = cur_;
    int words = 0;

    auto append = [&](const char* s, size_t n) {
      if (p + n + 1 >= sizeof(probe)) n = sizeof(probe) - p - 1; // truncate safely
      std::memcpy(probe + p, s, n); p += n;
      probe[p] = '\0';
    };

    for (;;) {
      if (!(cur_ < end_ && is_ident_start(*cur_))) break;
      const char* wbeg = cur_;
      advance(); // first char
      while (cur_ < end_ && is_ident_char(*cur_)) advance();
      const char* wend = cur_;

      if (words > 0) { if (p+1 < sizeof(probe)) probe[p++] = ' '; }
      append(wbeg, size_t(wend - wbeg));
      words++;

      const char *save2 = cur_;
      uint32_t sl2_line = line_, sl2_col = col_; bool sl2_bol = at_line_start_;
      skip_inline_space_only();
      if (!(cur_ < end_ && is_ident_start(*cur_)) || words >= 4) {
        cur_ = save2; line_ = sl2_line; col_ = sl2_col; at_line_start_ = sl2_bol; // restore spacing
        break;
      }
    }

    // Probe normalization + token.cpp flexible matching (handles underscores/spaces/case).
    Prim prim;
    if (from_flexible_spelling(probe, &prim)) {
      *out_prim = prim;
      *out_start = first;
      *out_end   = cur_;
      return true;
    }

    // Not a collapsible primitive — restore and report failure
    cur_ = save; line_ = save_line; col_ = save_col; at_line_start_ = save_at_bol;
    return false;
  }

  // After reading first ident char is available (peek is ident_start)
  Token scan_identifier_or_type() {
    const uint32_t start_line = line_;
    const uint32_t start_col  = col_;
    const char *tok_start_ptr = cur_;

    // First, try the primitive-type collapse (may consume multiple ident tokens).
    Prim prim; const char *s=nullptr; const char *e=nullptr;
    if (try_scan_primitive_collapse(&prim, &s, &e)) {
      return make_type_token(prim, s, size_t(e - s), start_line, start_col);
    }

    // Else: scan a single identifier
    advance(); // first char
    while (cur_ < end_ && is_ident_char(*cur_)) advance();
    const char *tok_end_ptr = cur_;
    std::string_view ident{tok_start_ptr, size_t(tok_end_ptr - tok_start_ptr)};

    // Single-word canonical primitive? (e.g., "int", "double", "wchar_t", "char8_t", "_Bool", "bool")
    if (from_spelling(std::string(ident).c_str(), &prim)) {
      return make_type_token(prim, tok_start_ptr, ident.size(), start_line, start_col);
    }

    // Keyword vs identifier
    if (is_keyword(ident)) {
      return make_simple(TokKind::Keyword, tok_start_ptr, ident.size(), start_line, start_col);
    }
    return make_simple(TokKind::Identifier, tok_start_ptr, ident.size(), start_line, start_col);
  }

  Token make_type_token(Prim prim, const char* s, size_t n, uint32_t ln, uint32_t col) {
    Token t;
    t.kind = TokKind::Type;
    t.lexeme = std::string_view{s, n};
    t.line = ln; t.column = col;
    t.prim = prim;
    return t;
  }

  void skip_inline_space_only() {
    while (cur_ < end_) {
      char c = *cur_;
      if (c == ' ' || c == '\t') { advance(); continue; }
      break;
    }
  }

  // ------------- numeric literals -------------

  Token scan_number() {
    const char *s = cur_;
    bool is_float = false;

    // Prefix: 0x / 0X (hex), 0b / 0B (binary), 0 (octal) — we’ll accept underscores in digits? (not in C/C++)
    if (*cur_ == '0' && (cur_+1 < end_)) {
      char n1 = *(cur_+1);
      if (n1 == 'x' || n1 == 'X') {
        advance(); advance();
        while (cur_ < end_ && std::isxdigit(unsigned(*cur_))) advance();
        consume_int_suffix();
        return make_simple(TokKind::Integer, s, size_t(cur_ - s), line_, col_);
      } else if (n1 == 'b' || n1 == 'B') {
        advance(); advance();
        while (cur_ < end_ && (*cur_=='0' || *cur_=='1')) advance();
        consume_int_suffix();
        return make_simple(TokKind::Integer, s, size_t(cur_ - s), line_, col_);
      }
    }

    // Decimal / octal, optional fraction / exponent
    bool saw_digits = false;
    while (cur_ < end_ && std::isdigit(unsigned(*cur_))) { advance(); saw_digits = true; }

    if (match('.')) {
      is_float = true;
      while (cur_ < end_ && std::isdigit(unsigned(*cur_))) advance();
    }

    // Exponent part (e/E for decimal, p/P for hex floats — we handle e/E for simplicity)
    if (cur_ < end_ && (*cur_=='e' || *cur_=='E')) {
      is_float = true;
      advance();
      if (cur_ < end_ && (*cur_=='+' || *cur_=='-')) advance();
      while (cur_ < end_ && std::isdigit(unsigned(*cur_))) advance();
    }

    // Suffixes
    if (is_float) {
      consume_float_suffix(); // f, F, l, L
      return make_simple(TokKind::Floating, s, size_t(cur_ - s), line_, col_);
    } else {
      consume_int_suffix(); // u/U, l/L, ll/LL, z/t (we’re permissive)
      return make_simple(TokKind::Integer, s, size_t(cur_ - s), line_, col_);
    }
  }

  void consume_int_suffix() {
    // Simple: accept combinations of u/U and l/L/ll/LL (and C++ size_t suffix 'z'/'Z' in some impls)
    const char *save;
    do {
      save = cur_;
      if (match('u') || match('U')) continue;
      if (match('l') || match('L')) {
        if (match('l') || match('L')) { /* LL */ }
        continue;
      }
      if (match('z') || match('Z') || match('t') || match('T')) continue;
    } while (cur_ != save);
  }

  void consume_float_suffix() {
    // f/F, l/L
    if (match('f') || match('F') || match('l') || match('L')) { /* ok */ }
  }

  // ------------- strings / chars -------------

  bool is_string_or_char_prefix() const {
    // Accept u8, u, U, L prefixes for string/char literals.
    if (*cur_ == '"' || *cur_ == '\'') return true;
    if (*cur_ == 'u' && (peek_next()=='8')) {
      const char c2 = (cur_+2 < end_) ? *(cur_+2) : '\0';
      return (c2=='"' || c2=='\'');
    }
    if ((*cur_=='u' || *cur_=='U' || *cur_=='L')) {
      const char c1 = peek_next();
      return (c1=='"' || c1=='\'');
    }
    return false;
  }

  Token scan_string_or_char() {
    const char *s = cur_;
    bool is_char = false;

    // Handle prefix
    if (*cur_=='u' && peek_next()=='8') { advance(); advance(); }
    else if (*cur_=='u' || *cur_=='U' || *cur_=='L') { advance(); }

    if (*cur_=='\'') is_char = true;
    if (*cur_=='\'' || *cur_=='"') {
      char quote = *cur_; advance();
      while (cur_ < end_) {
        char c = *cur_;
        if (c == '\\') { // escape
          advance();
          if (cur_ < end_) advance(); // skip escaped char (does not validate sequences)
          continue;
        }
        if (c == quote) { advance(); break; }
        if (c == '\n' || c == '\0') break; // unclosed
        advance();
      }
      return make_simple(is_char ? TokKind::CharLit : TokKind::StringLit,
                         s, size_t(cur_ - s), line_, col_);
    }

    // Shouldn’t reach here
    advance();
    return make_simple(TokKind::Unknown, s, 1, line_, col_);
  }

  // ------------- operators / punctuators -------------

  Token scan_operator_or_punct() {
    const char *s = cur_;
    auto two = [&](char a, char b){ return (cur_+1<end_ && *cur_==a && *(cur_+1)==b); };
    auto three = [&](char a, char b, char c){ return (cur_+2<end_ && *cur_==a && *(cur_+1)==b && *(cur_+2)==c); };

    // 3-char
    if (three('<','<','=')) { advance(); advance(); advance(); return make_simple(TokKind::Operator,s,3,line_,col_); }
    if (three('>','>','=')) { advance(); advance(); advance(); return make_simple(TokKind::Operator,s,3,line_,col_); }
    if (three('.','.','.')) { advance(); advance(); advance(); return make_simple(TokKind::Operator,s,3,line_,col_); }
    if (three(':',':','=')) { advance(); advance(); advance(); return make_simple(TokKind::Operator,s,3,line_,col_); } // GNU ext

    // 2-char
    if (two('+','+')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('-','-')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('-','>')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('+','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('-','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('*','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('/','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('%','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('&','&')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('|','|')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('=','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('!','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('<','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('>','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('<','<')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('>','>')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('&','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('|','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('^','=')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two(':',':')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('#','#')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('.','*')) { advance(); advance(); return make_simple(TokKind::Operator,s,2,line_,col_); }
    if (two('-','>') && (cur_+2<end_ && *(cur_+2)=='*')) { /* ->* */ }

    // 1-char
    const char c = *cur_;
    switch (c) {
      case '+': case '-': case '*': case '/': case '%':
      case '&': case '|': case '^': case '~': case '!':
      case '=': case '<': case '>': case '?': case ':':
      case ';': case ',': case '.':
      case '(': case ')': case '{': case '}': case '[': case ']':
      case '#':
        advance();
        return make_simple(TokKind::Operator, s, 1, line_, col_);
      default:
        break;
    }
    return make_simple(TokKind::Unknown, s, 1, line_, col_);
  }

private:
  const char *begin_;
  const char *cur_;
  const char *end_;
  uint32_t    line_;
  uint32_t    col_;
  bool        at_line_start_ = true;
};

}
