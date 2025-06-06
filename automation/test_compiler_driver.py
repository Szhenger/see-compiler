import unittest
import subprocess
import tempfile
import os

COMPILER_BIN = "build/compiler_driver"

class CompilerIntegrationTests(unittest.TestCase):

    def run_compiler(self, source_code):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as tmp:
            tmp.write(source_code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [COMPILER_BIN],
                stdin=open(tmp_path, 'r'),
                capture_output=True,
                text=True,
                timeout=5
            )
        finally:
            os.remove(tmp_path)

        return result.stdout.strip()

    def test_minimal_main(self):
        source = "int main(void) {}"
        output = self.run_compiler(source)
        self.assertIn("main", output)
        self.assertIn("{", output)
        self.assertIn("}", output)

    def test_main_with_return(self):
        source = "int main(void) { return 0; }"
        output = self.run_compiler(source)
        self.assertIn("return", output)
        self.assertIn("0", output)

    def test_hello_world(self):
        source = 'int main(void) { printf("Hello, world!\\n"); return 0; }'
        output = self.run_compiler(source)
        self.assertIn('printf', output)
        self.assertIn('"Hello, world!\\n"', output)
        self.assertIn('return', output)

if __name__ == '__main__':
    unittest.main()
