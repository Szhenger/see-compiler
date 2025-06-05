# Lexer Design Document

## Purpose

The lexer is responsible for performing lexical analysis on valid C source input. It converts raw character streams into a well-typed stream of tokens that reflect the syntax of the language. It is the first true stage of language comprehension in the SeeC pipeline.

## Token Responsibilities

- Classify tokens (keywords, identifiers, literals, symbols)
- Track token metadata (line, column, lexeme)
- Emit error tokens for invalid sequences

## Abstraction Contract

- **Input**: Raw character buffer (valid UTF-8 assumed)
- **Output**: `Token*` stream (with memory ownership noted)
- **Guarantees**:
  - No reordering
  - No backtracking beyond 1-token lookahead
  - Errors are embedded in-stream but do not halt scanning

## Token Types

| Token Type    | Examples       | Notes                    |
|---------------|----------------|--------------------------|
| `TOKEN_ID`     | `main`, `x`     | Includes variable names  |
| `TOKEN_INT`    | `42`, `0xFF`    | Integer literals         |
| `TOKEN_KW`     | `int`, `return` | Language keywords        |
| `TOKEN_SYM`    | `(`, `;`        | Symbols and punctuation  |
| `TOKEN_STR`    | `"Hello"`       | String literals          |
| `TOKEN_ERR`    | `??`            | Invalid lexemes          |

## Design Notes

- State-machine-based scanner
- Multi-character lookahead for complex tokens (e.g., `==`, `!=`) (eventually)
- Escapes and Unicode in strings are not yet handled

## Known Limitations

- No macro or preprocessor support yet
- UTF-8 validation is minimal (subject to extension)

## Testing

- C-based unit tests: `src/test_lexer.c`
- Python integration test: `tests/test_lexer_hello.py`
- Property-based fuzzing planned via Python harness
