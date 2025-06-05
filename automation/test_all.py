import subprocess
import sys
import os
import difflib

TESTS = [
    {
        "name": "lexer",
        "exec": "./build/test_lexer",
        "expected": "tests/lexer.expected"
    },
    {
        "name": "parser",
        "exec": "./build/test_parser",
        "expected": "tests/parser.expected"
    }
]

def run_test(test):
    print(f"Running {test['name']}...", end=' ')
    try:
        output = subprocess.check_output(test["exec"], stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print("❌ crashed")
        print(e.output)
        return False

    with open(test["expected"], "r") as f:
        expected = f.read()

    if output.strip() == expected.strip():
        print("✅ passed")
        return True
    else:
        print("❌ failed")
        diff = difflib.unified_diff(
            expected.splitlines(),
            output.splitlines(),
            fromfile="expected",
            tofile="actual",
            lineterm=""
        )
        print("\n".join(diff))
        return False

def main():
    all_passed = True
    for test in TESTS:
        passed = run_test(test)
        if not passed:
            all_passed = False
    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
