"""
test_config.py — proto-based frontend runner (Stage 5 replacement for conftest.py).

This module provides a SirRunner implementation backed by real protobuf
deserialisation of the .sir output file, replacing the Logger-scraping
approach in conftest.py once Stage 5 serialisation is implemented.

To activate:
  1. Compile sir.proto:
       protoc --python_out=. sir.proto
  2. Set SEECPP_FRONTEND_BINARY or build the project.
  3. Replace the SirRunner import in conftest.py with this module.

Run with:
    pytest test_config.py -v
"""

import os
import subprocess
import pytest

# ---------------------------------------------------------------------------
# Optional proto import — skip gracefully if bindings not yet generated.
# ---------------------------------------------------------------------------
try:
    import sir_pb2
    _PROTO_AVAILABLE = True
except ImportError:
    _PROTO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRONTEND_BINARY = os.environ.get(
    "SEECPP_FRONTEND_BINARY",
    os.path.join(os.path.dirname(__file__),
                 "../build/source/driver/frontend_driver"),
)
BINARY_TIMEOUT_S = 30

# ---------------------------------------------------------------------------
# Proto-backed SirRunner
# Implements the same interface as the Logger-scraping SirRunner in conftest.py
# so all existing tests work without modification once this is activated.
# ---------------------------------------------------------------------------

class SirRunner:
    """
    Drives the frontend_driver binary and parses the .sir proto output.

    Interface is intentionally identical to the SirRunner in conftest.py
    so that swapping implementations requires only changing the import.
    """

    def __init__(self, binary_path: str = FRONTEND_BINARY) -> None:
        self.binary_path = binary_path

    def run(self, onnx_path: str, sir_path: str) -> subprocess.CompletedProcess:
        """Run the binary and return the raw CompletedProcess (no assertions)."""
        return subprocess.run(
            [self.binary_path, onnx_path, sir_path],
            capture_output=True,
            text=True,
            timeout=BINARY_TIMEOUT_S,
        )

    def run_and_parse(self, onnx_path: str, sir_path: str) -> "sir_pb2.BlockProto":
        """
        Run the binary and deserialise the .sir output into a BlockProto.

        Raises:
            pytest.skip   — if proto bindings are not available
            AssertionError — if the binary exits non-zero
            AssertionError — if the .sir file was not written
        """
        if not _PROTO_AVAILABLE:
            pytest.skip(
                "sir_pb2 not found — run `protoc --python_out=. sir.proto` "
                "to generate Python bindings before using the proto runner"
            )

        result = self.run(onnx_path, sir_path)

        assert result.returncode == 0, (
            f"frontend_driver exited {result.returncode}\n"
            f"[stdout]\n{result.stdout}\n"
            f"[stderr]\n{result.stderr}"
        )

        assert os.path.isfile(sir_path), (
            f"frontend_driver exited 0 but did not write '{sir_path}' — "
            "check that Stage 5 serialisation is implemented"
        )

        block = sir_pb2.BlockProto()
        with open(sir_path, "rb") as f:
            block.ParseFromString(f.read())
        return block


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def runner() -> SirRunner:
    """
    Session-scoped SirRunner backed by proto deserialisation.

    Skips the entire session if:
      - the frontend_driver binary has not been built, or
      - the sir_pb2 proto bindings have not been generated.
    """
    if not os.path.isfile(FRONTEND_BINARY):
        pytest.skip(
            f"frontend_driver binary not found at '{FRONTEND_BINARY}'. "
            "Set SEECPP_FRONTEND_BINARY or build the project first."
        )

    if not _PROTO_AVAILABLE:
        pytest.skip(
            "sir_pb2 proto bindings not found. "
            "Run `protoc --python_out=. sir.proto` to generate them."
        )

    return SirRunner(FRONTEND_BINARY)