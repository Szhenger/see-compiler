# SeeComplier: A Beginner-Friendly C/C++ Compiler 

This is **SeeCompiler**, a readable and modular C/C++ compiler that translates a verified subset of the C programming language into a sequence of x86 assembly instructions. Designed for correctness, clarity and traceability, the compiler exposes each major compilation step with beginner-friendly source code and debug-friendly output.

At the core of SeeCompiler is `driver.c`, the management abstraction layer that runs the full compilation pipeline in a linear and inspectable process. This makes the project ideal for education, experimentation, and systems-level programming exploration.

## Driver

Internally, `driver.c` calls each compiler subroutine in sequence, validating results before advancing to the next procedure. We use clear diagnostic messages and structured output to aid learning and debugging. Each major data structure (tokens, AST, IR) is printed to the console, allowing users to see the transformation pipeline from source to assembly. Memory is fully freed after compilation, ensuring stability and leak-free operations across repeated runs.

The lexer is the first step of this compilation pipeline, which is housed in `lexer.c`. The purposw of this module is to convert raw C source code into a stream of tokens, each annotated with a category and lexeme (a literal representation).

## Lexer

Internally, `lexer.c` has the functionality to support tokenization:
* Primitive types (int, char, bool, string, etc.)
* Control flow keywords (if, else, while, for, etc.)
* Literals: integers, characters, strings
* Symbols: both single-character and multi-character (e.g., ==, ++, ->)
* Comments: skips both // and /* ... */ formats
* Identifiers: variable/function names, respecting C conventions
are all handled and analyzed by the module. 

Tokens are dynamically allocated and returned via the `tokenize()` function, which also emits a final `TOKEN_EOF` to signal the end of input. A corresponding `free_tokens()` function is provided to release all memory used by the token stream.

Next, the parser is the second step of our compilation, converting a stream of tokens into a tree-structured representation of the source code’s grammar: the Abstract Syntax Tree (AST). The code of which is contained in `parser.c`.

## Parser

Internally, the parser is powered by a `Parser` struct that maintains a token stream and cursor. It uses:
* `match()` and `advance()` utilities for stream control
* `parse_expression_with_precedence()` for left-to-right operator parsing
* Specialized routines for each syntactic construct (`parse_if()`, `parse_while()`, `parse_return()`, etc.)

This module transforms a flat token list into a deeply structured representation that guides semantic checks and backend code generation by implementing a recursive descent parser with full support for:
 Variable declarations and assignments
* Arithmetic and logical expressions with operator precedence
* Control flow: if, else, while, for
* Function calls and return statements.

The unit outputs an ASTNode tree rooted at the program’s main function and organizes ASTs using left/right children and statement lists. All statements are parsed into a unified `AST_STATEMENT_LIST` form to simplify traversal in later processes. 

# Build

TODO
