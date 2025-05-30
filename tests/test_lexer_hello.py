import subprocess

expected_output = [
    '[TOKEN_KEYWORD, "int"]',
    '[TOKEN_IDENTIFIER, "main"]',
    '[TOKEN_SYMBOL, "("]',
    '[TOKEN_KEYWORD, "void"]',
    '[TOKEN_SYMBOL, ")"]',
    '[TOKEN_SYMBOL, "{"]',
    '[TOKEN_IDENTIFIER, "printf"]',
    '[TOKEN_SYMBOL, "("]',
    '[TOKEN_STRING_LITERAL, "Hello, world!\\n"]',
    '[TOKEN_SYMBOL, ")"]',
    '[TOKEN_SYMBOL, ";"]',
    '[TOKEN_KEYWORD, "return"]',
    '[TOKEN_INTEGER_LITERAL, "0"]',
    '[TOKEN_SYMBOL, ";"]',
    '[TOKEN_SYMBOL, "}"]',
]

def run_lexer_test():
    result = subprocess.run(
        ["./build/seec"],
        capture_output=True,
        text=True
    )
    
    output_lines = result.stdout.strip().splitlines()

    for expected, actual in zip(expected_output, output_lines):
        assert expected == actual, f"Expected: {expected}, Got: {actual}"

    print("Lexer Hello World test passed!")

if __name__ == "__main__":
    run_lexer_test()
