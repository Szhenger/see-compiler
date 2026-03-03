"""
Integration tests for the SeeC++ frontend driver.

Run with:
    pytest test_frontend.py -v

Environment:
    SEECPP_FRONTEND_BINARY  path to the compiled frontend_driver binary
                            (default: ../build/source/driver/frontend_driver)
"""

import os
import subprocess
import pytest
import onnx
from onnx import helper, TensorProto, checker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRONTEND_BINARY = os.environ.get(
    "SEECPP_FRONTEND_BINARY",
    os.path.join(os.path.dirname(__file__), "../build/source/driver/frontend_driver"),
)
BINARY_TIMEOUT_S = 30  # seconds before we declare a hang

# ---------------------------------------------------------------------------
# ONNX model factories
# ---------------------------------------------------------------------------

def make_matmul_relu_model() -> onnx.ModelProto:
    """MatMul([2,4], [4,8]) -> [2,8], then Relu -> [2,8]."""
    nodes = [
        helper.make_node("MatMul", ["A", "B"], ["C"]),
        helper.make_node("Relu",   ["C"],       ["D"]),
    ]
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 8])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [2, 8])
    graph = helper.make_graph(nodes, "matmul-relu", [A, B], [D])
    model = helper.make_model(graph, producer_name="seecpp-test",
                              opset_imports=[helper.make_opsetid("", 17)])
    checker.check_model(model)
    return model


def make_conv_bn_relu_model() -> onnx.ModelProto:
    """Conv2D + BatchNorm + Relu — representative of a ResNet stem."""
    import numpy as np

    W    = helper.make_tensor("W",    TensorProto.FLOAT, [8, 3, 3, 3],
                               np.random.randn(8, 3, 3, 3).flatten().tolist())
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [8],
                                np.ones(8).tolist())
    bias  = helper.make_tensor("bias",  TensorProto.FLOAT, [8],
                                np.zeros(8).tolist())
    mean  = helper.make_tensor("mean",  TensorProto.FLOAT, [8],
                                np.zeros(8).tolist())
    var   = helper.make_tensor("var",   TensorProto.FLOAT, [8],
                                np.ones(8).tolist())

    nodes = [
        helper.make_node("Conv",              ["X", "W"],
                         ["conv_out"],
                         strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization", ["conv_out", "scale", "bias",
                                                "mean",    "var"],
                         ["bn_out"]),
        helper.make_node("Relu",              ["bn_out"], ["Y"]),
    ]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 8, 8])

    graph = helper.make_graph(nodes, "conv-bn-relu", [X], [Y],
                               initializer=[W, scale, bias, mean, var])
    model = helper.make_model(graph, producer_name="seecpp-test",
                               opset_imports=[helper.make_opsetid("", 17)])
    checker.check_model(model)
    return model


def make_unsupported_op_model() -> onnx.ModelProto:
    """Model containing an op the SeeC++ frontend does not yet handle."""
    node  = helper.make_node("LSTM", ["X", "W", "R"], ["Y"])
    X     = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4])
    W     = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 8, 4])
    R     = helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 8, 8])
    Y     = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 8])
    graph = helper.make_graph([node], "lstm-model", [X, W, R], [Y])
    # Skip checker — LSTM without full required inputs is structurally invalid
    # in ONNX but sufficient to test our unsupported-op handling.
    return helper.make_model(graph, producer_name="seecpp-test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_driver(onnx_path: str, sir_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [FRONTEND_BINARY, onnx_path, sir_path],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT_S,
    )


def dump_output(result: subprocess.CompletedProcess) -> None:
    """Print captured stdout/stderr — shown by pytest only on failure."""
    if result.stdout:
        print("[stdout]\n" + result.stdout)
    if result.stderr:
        print("[stderr]\n" + result.stderr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def onnx_path(tmp_path):
    return str(tmp_path / "model.onnx")

@pytest.fixture
def sir_path(tmp_path):
    return str(tmp_path / "model.sir")


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_driver_exits_zero_matmul_relu(self, onnx_path, sir_path):
        """Binary must return 0 for a well-formed MatMul+Relu model."""
        with open(onnx_path, "wb") as f:
            f.write(make_matmul_relu_model().SerializeToString())

        result = run_driver(onnx_path, sir_path)
        dump_output(result)
        assert result.returncode == 0, \
            f"Driver exited {result.returncode}; see stderr above"

    def test_driver_exits_zero_conv_bn_relu(self, onnx_path, sir_path):
        """Binary must return 0 for a Conv+BN+Relu stem model."""
        with open(onnx_path, "wb") as f:
            f.write(make_conv_bn_relu_model().SerializeToString())

        result = run_driver(onnx_path, sir_path)
        dump_output(result)
        assert result.returncode == 0, \
            f"Driver exited {result.returncode}; see stderr above"

    @pytest.mark.xfail(
        reason="SIR serialisation not yet implemented (Stage 5 stub)",
        strict=True,
    )
    def test_sir_file_is_written(self, onnx_path, sir_path):
        """Output .sir file must exist after a successful run."""
        with open(onnx_path, "wb") as f:
            f.write(make_matmul_relu_model().SerializeToString())
        run_driver(onnx_path, sir_path)
        assert os.path.exists(sir_path)


# ---------------------------------------------------------------------------
# Stderr / diagnostic content tests
# ---------------------------------------------------------------------------

class TestDiagnostics:

    def test_stderr_contains_pipeline_complete(self, onnx_path, sir_path):
        """Logger must emit the pipeline-complete message on success."""
        with open(onnx_path, "wb") as f:
            f.write(make_matmul_relu_model().SerializeToString())

        result = run_driver(onnx_path, sir_path)
        dump_output(result)
        combined = result.stdout + result.stderr
        assert "pipeline completed" in combined.lower(), \
            "Expected pipeline-complete message in output"

    def test_stderr_contains_op_count(self, onnx_path, sir_path):
        """Logger must report how many ops were ingested."""
        with open(onnx_path, "wb") as f:
            f.write(make_matmul_relu_model().SerializeToString())

        result = run_driver(onnx_path, sir_path)
        dump_output(result)
        combined = result.stdout + result.stderr
        assert "op" in combined.lower(), \
            "Expected op count in logger output"


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

class TestErrorPaths:

    def test_missing_binary_args_returns_nonzero(self):
        """Driver called with no arguments must exit non-zero."""
        result = subprocess.run(
            [FRONTEND_BINARY],
            capture_output=True, text=True, timeout=BINARY_TIMEOUT_S,
        )
        assert result.returncode != 0

    def test_nonexistent_onnx_file_returns_nonzero(self, sir_path):
        """Driver must fail gracefully when the input file does not exist."""
        result = run_driver("/tmp/does_not_exist_seecpp.onnx", sir_path)
        dump_output(result)
        assert result.returncode != 0

    def test_corrupt_protobuf_returns_nonzero(self, onnx_path, sir_path, tmp_path):
        """Driver must return non-zero when the .onnx file is not a valid protobuf."""
        with open(onnx_path, "wb") as f:
            f.write(b"\xFF\xFE garbage data that is not a protobuf \x00\x01")

        result = run_driver(onnx_path, sir_path)
        dump_output(result)
        assert result.returncode != 0

    def test_unsupported_op_returns_nonzero(self, onnx_path, sir_path):
        """Driver must return non-zero when the model contains an unsupported op."""
        with open(onnx_path, "wb") as f:
            f.write(make_unsupported_op_model().SerializeToString())

        result = run_driver(onnx_path, sir_path)
        dump_output(result)
        assert result.returncode != 0

    def test_error_message_in_stderr(self, sir_path):
        """When ingestion fails, an [ERROR] line must appear in stderr."""
        result = run_driver("/tmp/does_not_exist_seecpp.onnx", sir_path)
        dump_output(result)
        assert "[ERROR]" in result.stderr, \
            "Expected [ERROR] tag in stderr on ingestion failure"