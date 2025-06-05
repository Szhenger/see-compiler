# SeeC

This is **SeeC**, a modular and self-contained C/C++ compiler written in C that translates a verified subset of the C programming language into x86 assembly instructions. It is designed to be testable, extensible, and usable as a personal compilation tool for systems software.

> **Current Status**: Driver, Lexer, and Parser implemented and verified against a standard `Hello, world!` program.

---

## Purpose

SeeC is not just a toy — it’s a compiler I am building for personal use. I want a system I understand completely, can debug from the ground up, and can rely on when writing low-level software. SeeC provides me complete control of the compilation stack, from source to assembly.

The long-term goal is to use SeeC to compile my own utilities, OS-level experiments, and eventually small C libraries — with confidence in every phase of the pipeline.

---

## Compilation Pipeline (Full Plan)

SeeC follows the classic compiler architecture, designed with strict modularity and long-term extensibility in mind:

| Step                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Driver**             | Manages the full pipeline from input file to binary output                  |
| **Lexer**              | Converts raw source code into a token stream                                |
| **Parser**             | Constructs an abstract syntax tree (AST) from the token stream              |
| **AST Optimizer**      | (Planned) Simplifies or desugars AST nodes for canonical IR generation      |
| **Semantic Analysis**  | Validates type rules, scope, variable lifetimes, and symbol bindings        |
| **IR Generation**      | Converts AST to a typed intermediate representation for backend translation |
| **x86 Code Generator** | Emits verified x86 assembly from IR (focus: readability and safety)         |
| **Assembler/Linker**   | Integrates with `nasm` or similar to produce final executable               |

The early milestones (Lexer -> Parser) are completed. Later stages will be rolled out iteratively, each with independent testing and documentation.

---

## Project Layout

```bash
seec/
├── src/                  # Core compiler logic
│   ├── driver.c
│   ├── lexer.c
│   ├── parser.c
│   ├── test_*.c
├── include/              # Interfaces and data types
│   ├── lexer.h
│   ├── parser.h
│   ├── token.h
│   └── ast.h
├── utils/                # AST and token utilities
│   ├── token.c
│   └── ast.c
├── tests/                # Python-based and C-based test harnesses
│   ├── test_lexer_hello.py
│   ├── test_all.py
│   └── test_parser.py
├── docs/                 # Design contracts and dev notes
├── build/                # Compiled binaries
└── Makefile              # Build and test automation
