# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -g -Iinclude

# Source files
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests

# Targets
MAIN_SRC = $(SRC_DIR)/main.c
LEXER_TEST_SRC = $(SRC_DIR)/lexer_test.c
LEXER_SRC = $(SRC_DIR)/lexer.c

# Output binaries
MAIN_BIN = $(BUILD_DIR)/seec
LEXER_TEST_BIN = $(BUILD_DIR)/lexer_test

.PHONY: all clean

all: $(MAIN_BIN)

$(MAIN_BIN): $(MAIN_SRC) $(LEXER_SRC)
	$(CC) $(CFLAGS) -o $@ $^

lexer_test: $(LEXER_TEST_SRC) $(LEXER_SRC)
	$(CC) $(CFLAGS) -o $(LEXER_TEST_BIN) $^

clean:
	rm -f $(BUILD_DIR)/*
