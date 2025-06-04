# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -g -Iinclude

# Source files
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

# Targets
DRIVER_SRC = $(SRC_DIR)/driver.c
LEXER_SRC = $(SRC_DIR)/lexer.c
LEXER_TEST_SRC = $(SRC_DIR)/test_lexer.c
PARSER_SRC = $(SRC_DIR)/parser.c
PARSER_TEST_SRC = $(SRC_DIR)/test_parser.c

# Output binaries
MAIN_BIN = $(BUILD_DIR)/seec
LEXER_TEST_BIN = $(BUILD_DIR)/lexer_test
PARSER_TEST_BIN = $(BUILD_DIR)/parser_test

.PHONY: all clean

# Default target
all: $(MAIN_BIN)

$(MAIN_BIN): $(DRIVER_SRC) $(LEXER_SRC) src/token.c
	$(CC) $(CFLAGS) -o $@ $^

# Build lexer test
lexer_test: $(LEXER_TEST_SRC) $(LEXER_SRC) src/token.c
	$(CC) $(CFLAGS) -o $(LEXER_TEST_BIN) $^

# Clean build files
clean:
	rm -f $(BUILD_DIR)/*
