# Compiler and Flags
CC = gcc
CFLAGS = -Wall -Wextra -g -Iinclude

# Directories
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = test
UTIL_DIR = util

# Source files
DRIVER_SRC = $(SRC_DIR)/driver.c
LEXER_SRC = $(SRC_DIR)/lexer.c
PARSER_SRC = $(SRC_DIR)/parser.c
SEMANTIC_SRC = $(SRC_DIR)/semantic.c
IR_SRC = $(SRC_DIR)/ir.c
CODEGEN_SRC = $(SRC_DIR)/codegen.c

# Utility source files
AST_UTIL = $(UTIL_DIR)/ast.c
TOKEN_UTIL = $(UTIL_DIR)/token.c

# Test source files
LEXER_TEST = $(TEST_DIR)/test_lexer.c
PARSER_TEST = $(TEST_DIR)/test_parser.c
SEMANTIC_TEST = $(TEST_DIR)/test_semantic.c
CODEGEN_TEST = $(TEST_DIR)/test_codegen.c

# Output binaries
DRIVER_BIN = $(BUILD_DIR)/seec
LEXER_TEST_BIN = $(BUILD_DIR)/lexer_test
PARSER_TEST_BIN = $(BUILD_DIR)/parser_test
SEMANTIC_TEST_BIN = $(BUILD_DIR)/semantic_test
CODEGEN_TEST_BIN = $(BUILD_DIR)/codegen_test

.PHONY: all clean test

# Ensure build directory exists
$(shell mkdir -p $(BUILD_DIR))

# Default target
all: $(DRIVER_BIN)

$(DRIVER_BIN): $(DRIVER_SRC) $(LEXER_SRC) $(PARSER_SRC) $(SEMANTIC_SRC) $(IR_SRC) $(CODEGEN_SRC) $(AST_UTIL) $(TOKEN_UTIL)
	$(CC) $(CFLAGS) -o $@ $^

# Build lexer test
lexer_test: $(LEXER_TEST) $(LEXER_SRC) $(TOKEN_UTIL)
	$(CC) $(CFLAGS) -o $(LEXER_TEST_BIN) $^

# Build parser test
parser_test: $(PARSER_TEST) $(AST_UTIL) $(LEXER_SRC) $(PARSER_SRC) $(TOKEN_UTIL)
	$(CC) $(CFLAGS) -o $(PARSER_TEST_BIN) $^

# Build semantic test
semantic_test: $(SEMANTIC_TEST) $(AST_UTIL) $(LEXER_SRC) $(PARSER_SRC) $(SEMANTIC_SRC) $(TOKEN_UTIL) 
	$(CC) $(CFLAGS) -o $(SEMANTIC_TEST_BIN) $^

# Build codegen test
codegen_test: $(CODEGEN_TEST) $(AST_UTIL) $(LEXER_SRC) $(PARSER_SRC) $(SEMANTIC_SRC) $(IR_SRC) $(CODEGEN_SRC) $(TOKEN_UTIL)
	$(CC) $(CFLAGS) -o $(CODEGEN_TEST_BIN) $^

# Run all tests
test: lexer_test parser_test semantic_test codegen_test
	@echo "Running lexer_test..."
	@$(LEXER_TEST_BIN)
	@echo "Running parser_test..."
	@$(PARSER_TEST_BIN)
	@echo "Running semantic_test..."
	@$(SEMANTIC_TEST_BIN)
	@echo "Running codegen_test..."
	@$(CODEGEN_TEST_BIN)
	@echo "All tests passed."

# Clean build files
clean:
	rm -rf $(BUILD_DIR)/*

