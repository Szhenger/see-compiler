import subprocess
import unittest

class TestLexerHelloWorld(unittest.TestCase):
    def setUp(self):
        self.executable = "./build/seec"
        self.expected_output = [
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
            '[TOKEN_EOF, ""]'
        ]

    def run_lexer(self):
        result = subprocess.run(
            [self.executable],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().splitlines()

    def test_hello_world_tokens(self):
        actual_output = self.run_lexer()
        self.assertEqual(len(actual_output), len(self.expected_output), "Token count mismatch.")

        for i, (expected, actual) in enumerate(zip(self.expected_output, actual_output), 1):
            self.assertEqual(
                actual, expected,
                f"Mismatch at token {i}: expected {expected}, got {actual}"
            )

if __name__ == "__main__":
    unittest.main()

