# SeeComplier: A Beginner-Friendly C/C++ Compiler 

This is **SeeCompiler**, a readable and modular C/C++ compiler that translates a verified subset of the C programming language into a sequence of x86 assembly instructions. Designed for correctness, clarity and traceability, the compiler exposes each major compilation step with beginner-friendly source code and debug-friendly output.

At the core of SeeCompiler is `driver.c`, the management abstraction layer that runs the full compilation pipeline in a linear and inspectable process. This makes the project ideal for education, experimentation, and systems-level programming exploration.

## Driver

Internally, `driver.c` calls each compiler subroutine in sequence, validating results before advancing to the next procedure. We use clear diagnostic messages and structured output to aid learning and debugging. Each major data structure (tokens, AST, IR) is printed to the console, allowing users to see the transformation pipeline from source to assembly. Memory is fully freed after compilation, ensuring stability and leak-free operations across repeated runs.

Firstly, the lexer begans this compilation pipeline, which is housed in `lexer.c`. The purposw of this module is to convert raw C source code into a stream of tokens, each annotated with a category and lexeme (a literal representation).

## Lexer

Internally, `lexer.c` has the functionality to support tokenization:
* Primitive types (`int`, `char`, `bool`, `string`, etc.)
* Control flow keywords (`if`, `else`, `while`, `for`, etc.)
* Literals: integers, characters, strings
* Symbols: both single-character and multi-character (e.g., `==`, `++`, `->`)
* Comments: skips both `//` and `/* ... */` formats
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

The semantic analyzer is the third stage of compilation in `semantic.c`, responsible for enforcing language rules beyond syntax. It validates that:
* All variables are declared before use
* No redeclarations exist in the same scope
* Expressions are well-formed
* Return statements are correctly structured
* Control flow conditions are semantically valid.

## Semantic Analysis

Internally, this layer runs a simple compile-time execution environment, performing symbol tracking, scope validation, and structural checks of statements and expressions. 

Symbol Table Management:
* Maintains a linked-list-based symbol table for tracking variable declarations
* Implements `add_symbol()`, `symbol_exists()`, and `clear_symbols()` for symbol lifecycle
* Uses a single-scope environment (flat, no shadowing), suitable for validating simple `main`-only programs.

Expression Validation:
* Checks that identifiers are declared before use (`AST_IDENTIFIER`)
* Recursively validates binary expressions (`AST_BINARY_OP`) and function calls
* Reports semantic errors for undeclared variables and malformed expressions.

Statement-Level Semantics:
* `AST_DECLARATION`: ensures identifiers are unique and recorded in the symbol table
* `AST_ASSIGNMENT`: ensures LHS is a declared identifier and RHS is a valid expression
* `AST_RETURN_STMT`: requires a return value and checks that it’s semantically valid
* `AST_STATEMENT_LIST`: flattens and validates sequential statements recursively.

Control Flow Validation:
* `AST_IF_STMT`: verifies condition expression and both branches (then/else)
* `AST_WHILE_LOOP`: verifies condition and loop body
* `AST_FOR_LOOP`: validates initializer (`stmt`), condition (`expr`), step (`stmt`) in structured order.

Function-Level Verification:
* Entry point must be a function with `type == AST_FUNCTION_DEF` and `value == "main"`
* Entire function body is validated as a statement list.

Error Reporting:
* Use-before-declaration
* Redeclarations
* Missing return values
* Invalid constructs in branches or expressions.

AST Traversal Strategy:
* The analysis is top-down and recursive, using `switch` dispatch on AST node types
* Expression analysis is composed within statement analysis, composing semantics hierarchically.

Next, the IR generation phase serves as a lowering step, translating validated AST nodes into a flat, linear sequence of instructions. This IR abstracts away high-level language syntax into stack-based, pseudo-assembly operations, enabling backend code generation to be cleanly decoupled from AST shape.

## Intermediate Representation

Internally, this module converts the AST into a linear, stack-based Intermediate Representation (IR)—a simplified pseudo-assembly language that decouples high-level syntax from backend code generation. 

Translates each AST Node into a chain of IRInstr structs (type, arg, next) i.e. it supports core operations:
* Memory: `IR_DECL`, `IR_LOAD`, `IR_STORE`
* Literals & Stack: `IR_PUSH`
* Arithmetic: `IR_ADD`, `IR_SUB`, `IR_MUL`, `IR_DIV`
* Control Flow: `IR_LABEL`, `IR_JUMP`, `IR_JUMP_IF_ZERO`, `IR_CMP`
* Functions: `IR_CALL`, `IR_RET`.

High-level flow (e.g., if, while, for) is lowered into labeled jumps:
* `if` emits conditional branches to `else_LABEL` / `endif_LABEL`
* `while` emits looping conditions with `JUMP_IF_ZERO`
* `for` handles init/test/step with generated blocks and labels.

Lastly, the x86 code generation module emits Intel-style x86-64 assembly from the compiler’s intermediate representation (IR). It uses a stack-based model and generates correct function prologues, epilogues, and instruction sequences for expressions, memory access, and control flow.

## x86 Code Generation

Internally, this module converts the linear IR instruction stream into x86-64 assembly instructions, using a stack-based calling convention and Intel syntax by handling:

Variable Layout
* Maintains a linked `VarEntry` table mapping variable names to `rbp`-relative offsets.
* Offsets are aligned at 8 bytes, growing downward from `rbp`.

Code Structure
* Emits standard function prologue and epilogue around main.

Instruction Emission
* Each IR type (`IR_PUSH`, `IR_ADD`, `IR_LOAD`, etc.) maps to corresponding assembly snippets.
* Binary ops (`+`, `-`, `*`, `/`) are translated into stack-based computations using `rax`/`rbx`.
* Control flow (e.g., `if`, `while`, `for`) is handled via jump labels and conditional branches.
* Strings are stored in `.rodata`, labeled `.LCn`, and referenced using `lea rax, label`.

String Handling
* Escapes quotes, backslashes, and newlines
* Allocates each unique string as `.string` in a separate `.rodata` section

## Build

Prerequisites
* C99-compatible compiler (e.g., `clang`, `gcc`)
* Unix-like environment (Linux/macOS recommended)
* `make` utility

To build the compiler, simply run:

```bash
make
