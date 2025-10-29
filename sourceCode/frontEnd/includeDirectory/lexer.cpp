#pragma once
#include <cstdint>
#include <cstddef>
#include <string_view>

#include "token.hpp" // for see::Prim

namespace see {

enum class TokKind : std::uint16_t {
  End,
  Identifier,
  Keyword,       // non-primitive keywords (if/else/return/struct/…)
  Type,          // canonical primitive type (payload: Prim)
  Integer,
  Floating,
  CharLit,
  StringLit,
  Preprocessor,  // from '#' to end-of-line
  Operator,      // operators & punctuators
  Unknown
};

struct Token {
  TokKind          kind{TokKind::Unknown};
  std::string_view lexeme{};   // view into the source buffer
  std::uint32_t    line{0};    // 1-based
  std::uint32_t    column{0};  // 1-based
  // Valid iff kind==TokKind::Type
  Prim             prim{Prim::Void};
};

// A single-pass lexer that produces a token stream over a source buffer.
// Design: zero allocations; tokens hold string_views into the original text.
class Lexer {
public:
  Lexer(const char* src, std::size_t len);

  // Produce the next token (TokKind::End when finished).
  Token next();

private:
  // --- minimal character stream state (must match lexer.cpp) ---
  const char* begin_{nullptr};
  const char* cur_{nullptr};
  const char* end_{nullptr};
  std::uint32_t line_{1};
  std::uint32_t col_{1};
  bool at_line_start_{true};

  // Helpers declared here so out-of-line defs in lexer.cpp can access them.
  bool eof() const;
  char peek() const;
  char peek_next() const;
  void advance();
  bool match(char c);

  static bool is_ident_start(char c);
  static bool is_ident_char(char c);

  // Skipping / scanning helpers used by next()
  void skip_ws_and_comments();
  void skip_inline_space_only();

  // Token constructors
  static Token make_simple(TokKind k, const char* s, std::size_t n,
                           std::uint32_t ln, std::uint32_t col);

  // Scanners
  Token scan_identifier_or_type();
  Token scan_number();
  void  consume_int_suffix();
  void  consume_float_suffix();
  bool  is_string_or_char_prefix() const;
  Token scan_string_or_char();
  Token scan_operator_or_punct();

  // Primitive type collapsing (uses token.cpp flexible spelling)
  bool try_scan_primitive_collapse(Prim* out_prim, const char** out_start, const char** out_end);

  // Keyword probe (non-primitive keywords)
  static bool is_keyword(std::string_view s);

  // Type token helper
  Token make_type_token(Prim prim, const char* s, std::size_t n,
                        std::uint32_t ln, std::uint32_t col);
};

} // namespace see
