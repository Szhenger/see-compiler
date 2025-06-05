import subprocess
import unittest

class TestDriverHelloWorld(unittest.TestCase):
    def setUp(self):
        self.executable = "./build/seec"
        self.expected_output = [
            "Function: main",
            "  Call: printf",
            '    Arg: Hello, world!\\n',
            "  Return: 0"
        ]

    def run_driver(self):
        result = subprocess.run(
            [self.executable],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, "Driver exited with non-zero status")
        return result.stdout.strip().splitlines()

    def test_driver_output(self):
        actual_output = self.run_driver()
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
