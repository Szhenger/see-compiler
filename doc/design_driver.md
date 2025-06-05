# Driver Design Document

## Purpose

The driver is the central processing unit of the SeeC compilation pipeline. It receives input source files, invokes the lexer and parser in sequence, manages error states, and produces an intermediate output suitable for the next compiler stage (eventually, IR -> codegen).

## Responsibilities

- Validate input arguments (e.g., file paths)
- Open and manage input streams
- Invoke lexer and collect token stream
- Pass tokens to parser
- Report diagnostics to user
- Manage exit codes and system-level integration

## Abstraction Boundaries

- Inputs: `char *` input filenames (eventually)
- Outputs: status codes, logs, or ASTs (eventually)
- Depends on: `lexer.h`, `parser.h`

## Design Notes

- Uses standard I/O and assumes UTF-8 encoding
- Current design avoids memory-mapped files for portability
- All errors are printed to `stderr` for predictability

## Future Considerations

- Compiler flags and optimization levels
- Output file routing and intermediate representations
- Compilation mode (interpret, compile, emit IR)

## Testing

- Covered in `tests/test_driver.c`
- Verifies end-to-end behavior for basic files like `hello.c`
