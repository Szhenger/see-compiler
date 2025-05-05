CC = gcc
CFLAGS = -Wall -Wextra -Iinclude
SRC = src/main.c
OUT = build/seec

all: $(OUT)

$(OUT): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)
