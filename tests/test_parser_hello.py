import subprocess
import unittest

class TestParserOutput(unittest.TestCase):

    def test_hello_world_ast(self):
        """Runs test_parser executable and checks AST output"""
        result = subprocess.run(
            ["./build/test_parser"],  # Adjust path if needed
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0, "C test_parser did not exit cleanly")

        output = result.stdout.strip().splitlines()
        expected_output = [
            "Function: main",
            'Call: printf with arg: "Hello, world!\\n"',
            "Return: 0"
        ]

        for expected, actual in zip(expected_output, output):
            self.assertEqual(actual.strip(), expected, f"Expected '{expected}', got '{actual}'")

if __name__ == "__main__":
    unittest.main()
