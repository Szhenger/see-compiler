import subprocess
import unittest

class TestParserHelloWorld(unittest.TestCase):
    def setUp(self):
        self.executable = "./build/test_parser"
        self.expected_output = [
            "Function: main",
            'Call: printf with arg: Hello, world!\\n',
            "Return: 0"
        ]

    def run_parser(self):
        result = subprocess.run(
            [self.executable],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, "Parser executable did not exit cleanly")
        return result.stdout.strip().splitlines()

    def test_ast_structure(self):
        actual_output = self.run_parser()
        self.assertEqual(
            len(actual_output), len(self.expected_output),
            f"Expected {len(self.expected_output)} lines, got {len(actual_output)}."
        )

        for i, (expected, actual) in enumerate(zip(self.expected_output, actual_output), 1):
            self.assertEqual(
                actual.strip(), expected,
                f"Line {i} mismatch:\nExpected: {expected}\nGot     : {actual.strip()}"
            )

if __name__ == "__main__":
    unittest.main()

