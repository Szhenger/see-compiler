import unittest
import subprocess
import tempfile
import os

class TestCompilerErrorHandling(unittest.TestCase):
    COMPILER_BIN = "./build/seec"
  
    def run_compiler(self, source_code: str):
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as tmp_file:
            tmp_file.write(source_code)
            tmp_file_path = tmp_file.name
        try:
            result = subprocess.run(
                [COMPILER_BINARY, tmp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stdout, result.stderr, result.returncode
        finally:
            os.unlink(tmp_file_path)

    def test_syntax_error_missing_paren(self):
        source = "int main( { return 0; }"  # missing closing parenthesis
        stdout, stderr, returncode = self.run_compiler(source)
        self.assertNotEqual(returncode, 0)
        self.assertIn("Parsing failed", stderr)

    def test_unterminated_string_literal(self):
        source = 'int main() { printf("Hello;'  # missing ending quote
        stdout, stderr, returncode = self.run_compiler(source)
        self.assertNotEqual(returncode, 0)
        self.assertIn("Lexing failed", stderr)

    def test_unknown_token(self):
        source = 'int main() { int $a = 5; return 0; }'  # invalid token: '$'
        stdout, stderr, returncode = self.run_compiler(source)
        self.assertNotEqual(returncode, 0)
        self.assertIn("Lexing failed", stderr)

    def test_empty_source(self):
        source = ''
        stdout, stderr, returncode = self.run_compiler(source)
        self.assertNotEqual(returncode, 0)
        self.assertIn("Lexing failed", stderr)

    def test_missing_return_type(self):
        source = 'main() { return 0; }'  # missing return type 'int'
        stdout, stderr, returncode = self.run_compiler(source)
        self.assertNotEqual(returncode, 0)
        self.assertIn("Parsing failed", stderr)

    # Add more error cases here as needed

if __name__ == "__main__":
    unittest.main()
