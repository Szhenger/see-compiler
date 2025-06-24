# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -g -Iinclude

# Source files
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = test
UTIL_DIR = utils


# Targets
DRIVER_SRC = $(SRC_DIR)/driver.c
LEXER_SRC = $(SRC_DIR)/lexer.c
LEXER_TEST = $(TEST_DIR)/test_lexer.c
PARSER_SRC = $(SRC_DIR)/parser.c
PARSER_TEST = $(TEST_DIR)/test_parser.c
SEMANTIC_SRC = $(SRC_DIR)/semantic.c
SEMANTIC_TEST = $(TEST_DIR)/test_semantic.c
AST_UTIL = $(UTIL_DIR)/ast.c
TOKEN_UTIL = $(UTIL_DIR)/token.c

# Output binaries
DRIVER_BIN = $(BUILD_DIR)/seec
LEXER_TEST_BIN = $(BUILD_DIR)/lexer_test
PARSER_TEST_BIN = $(BUILD_DIR)/parser_test
SEMANTIC_TEST_BIN = $(BUILD_DIR)/semantic_test

.PHONY: all clean

# Default target
all: $(DRIVER_BIN)

$(DRIVER_BIN): $(DRIVER_SRC) $(LEXER_SRC) $(TOKEN_UTIL)
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

# Clean build files
clean:
	rm -f $(BUILD_DIR)/*
