#ifndef LEXER_HPP
#define LEXER_HPP

#include <string>
#include <vector>
#include "token.hpp"

struct Lexer
{
    std::string source;

    size_t index;      // current byte index into `source`
    int line;          // current line (1-based)
    int column;        // current column (1-based)

    explicit Lexer(const std::string& input);

    Token next_token();

    Token peek_token();

    bool is_eof() const;

    std::vector<Token> tokenize_all();

private:

    char current_char() const;
    char peek_char(int ahead = 1) const;
    void advance(int count = 1);

    void skip_whitespace();
    void skip_comment(); // handles // and /* */ comments

    Token lex_identifier_or_keyword();
    Token lex_number_literal();
    Token lex_string_literal();
    Token lex_char_literal();

    Token lex_operator();
    Token lex_punctuation();

    Token make_token(TokenKind kind, const std::string& text);
};

#endif // LEXER_HPP
