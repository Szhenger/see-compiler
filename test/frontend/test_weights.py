"""
test_weight_buffer.py — component tests for ONNX initializer ingestion
                        and WeightBuffer integration.

Verifies that ONNX initializers (weights, biases) are correctly:
  1. Translated into sc_high.constant ops in the SIR graph
  2. Stored in the global WeightBuffer with correct shape metadata
  3. Accessible by downstream ops via the symbol table

Run with:
    pytest test_weight_buffer.py -v
"""

import numpy as np
import pytest
import onnx
from onnx import helper, TensorProto, checker, numpy_helper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opset(version: int = 17):
    return [helper.make_opsetid("", version)]


def _write_model(path: str, model: onnx.ModelProto) -> None:
    try:
        checker.check_model(model)
    except checker.ValidationError as e:
        raise ValueError(f"Test fixture is not valid ONNX: {e}") from e
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def _find_constant(summary, weight_name: str):
    """Return the OpSummary for a sc_high.constant op whose result id == weight_name."""
    for op in summary.ops:
        if "constant" in op.mnemonic.lower():
            for r in op.results:
                if r.value_id == weight_name:
                    return op
    return None


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestConstantIngestion:

    def test_single_initializer_becomes_constant_op(self, runner, tmp_path):
        """A single ONNX initializer must produce one sc_high.constant op."""
        onnx_file = str(tmp_path / "const.onnx")
        sir_file  = str(tmp_path / "const.sir")

        W_data   = np.random.randn(3, 3).astype(np.float32)
        W_tensor = numpy_helper.from_array(W_data, name="W")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "W"], ["Y"])],
            "single-const", [X], [Y], initializer=[W_tensor])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        w_op = _find_constant(summary, "W")
        assert w_op is not None, \
            "Expected a sc_high.constant op for initializer 'W' in SIR output"

    def test_initializer_shape_preserved(self, runner, tmp_path):
        """The sc_high.constant result must carry the initializer's exact shape."""
        onnx_file = str(tmp_path / "shape.onnx")
        sir_file  = str(tmp_path / "shape.sir")

        W_data   = np.random.randn(3, 3).astype(np.float32)
        W_tensor = numpy_helper.from_array(W_data, name="W")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "W"], ["Y"])],
            "shape-const", [X], [Y], initializer=[W_tensor])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        w_op = _find_constant(summary, "W")
        assert w_op is not None, "sc_high.constant for 'W' not found"

        w_result = next(r for r in w_op.results if r.value_id == "W")
        assert w_result.dims == [3, 3], \
            f"Expected shape [3,3], got {w_result.dims}"

    @pytest.mark.parametrize("shape", [
        [8],             # bias vector
        [64, 3, 3, 3],   # Conv2D filter (ResNet stem)
        [512, 256],      # Gemm weight matrix
    ])
    def test_various_initializer_shapes(self, runner, tmp_path, shape):
        """Initializers of different ranks must all resolve to correct shapes."""
        onnx_file = str(tmp_path / "varshape.onnx")
        sir_file  = str(tmp_path / "varshape.sir")

        # Build a minimal graph: Reshape input to match W then Add.
        # We use a MatMul for rank-2, Add for rank-1 and rank-4 with broadcast.
        W_data   = np.zeros(shape, dtype=np.float32)
        W_tensor = numpy_helper.from_array(W_data, name="W")

        # Make a flat input that we reshape — keeps the ONNX graph valid
        # regardless of W's rank.
        flat = int(np.prod(shape))
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [flat])

        shape_const = numpy_helper.from_array(
            np.array(shape, dtype=np.int64), name="target_shape")
        X_reshaped = helper.make_tensor_value_info("Xr", TensorProto.FLOAT, shape)
        Y          = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, shape)

        nodes = [
            helper.make_node("Reshape", ["X", "target_shape"], ["Xr"]),
            helper.make_node("Add",     ["Xr", "W"],           ["Y"]),
        ]
        graph = helper.make_graph(
            nodes, "varshape", [X], [Y],
            initializer=[W_tensor, shape_const])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        w_op = _find_constant(summary, "W")
        assert w_op is not None, \
            f"sc_high.constant for 'W' not found (shape={shape})"
        w_result = next(r for r in w_op.results if r.value_id == "W")
        assert w_result.dims == shape, \
            f"Shape mismatch: expected {shape}, got {w_result.dims}"

    def test_multiple_initializers_all_ingested(self, runner, tmp_path):
        """All initializers in the graph must each produce a constant op."""
        onnx_file = str(tmp_path / "multi.onnx")
        sir_file  = str(tmp_path / "multi.sir")

        W1 = numpy_helper.from_array(np.ones((4, 4),  dtype=np.float32), "W1")
        W2 = numpy_helper.from_array(np.zeros((4,),   dtype=np.float32), "W2")
        W3 = numpy_helper.from_array(np.eye(4,        dtype=np.float32), "W3")

        X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [4, 4])
        Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [4, 4])

        # X @ W1 + W2, then Add W3
        nodes = [
            helper.make_node("Gemm", ["X",  "W1", "W2"], ["T"]),
            helper.make_node("Add",  ["T",  "W3"],        ["Y"]),
        ]
        graph = helper.make_graph(nodes, "multi", [X], [Y],
                                   initializer=[W1, W2, W3])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        for name, expected_shape in [("W1", [4, 4]), ("W2", [4]), ("W3", [4, 4])]:
            op = _find_constant(summary, name)
            assert op is not None, \
                f"sc_high.constant for '{name}' not found in SIR"
            r = next(res for res in op.results if res.value_id == name)
            assert r.dims == expected_shape, \
                f"'{name}': expected {expected_shape}, got {r.dims}"

    def test_weight_element_count_logged(self, runner, tmp_path):
        """Logger must report the number of ops ingested, confirming weights
        were processed (proxy for WeightBuffer.add() being called)."""
        onnx_file = str(tmp_path / "elemcount.onnx")
        sir_file  = str(tmp_path / "elemcount.sir")

        W = numpy_helper.from_array(np.ones((2, 4), dtype=np.float32), "W")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "W"], ["Y"])],
            "elemcount", [X], [Y], initializer=[W])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        # num_ops includes the constant op for W plus the Add op — must be >= 2.
        assert summary.num_ops >= 2, \
            f"Expected at least 2 ops (constant + Add), got {summary.num_ops}"


# ---------------------------------------------------------------------------
# raw_data encoding (the common real-world case)
# ---------------------------------------------------------------------------

class TestRawDataEncoding:

    def test_raw_data_initializer_shape_correct(self, runner, tmp_path):
        """Weights written via raw_data (numpy_helper default) must ingest correctly.

        Most ONNX exporters (PyTorch, TF) use raw_data rather than float_data.
        numpy_helper.from_array produces raw_data encoding by default.
        """
        onnx_file = str(tmp_path / "raw.onnx")
        sir_file  = str(tmp_path / "raw.sir")

        # numpy_helper.from_array always writes raw_data for float tensors.
        W_data   = np.random.randn(8, 3, 3, 3).astype(np.float32)
        W_tensor = numpy_helper.from_array(W_data, name="W")
        assert len(W_tensor.raw_data) > 0,   "fixture: expected raw_data encoding"
        assert len(W_tensor.float_data) == 0, "fixture: float_data should be empty"

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 5, 5])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        node = helper.make_node("Conv", ["X", "W"], ["Y"],
                                strides=[1, 1], pads=[0, 0, 0, 0])
        graph = helper.make_graph([node], "raw-conv", [X], [Y],
                                   initializer=[W_tensor])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        w_op = _find_constant(summary, "W")
        assert w_op is not None, \
            "Conv filter initializer 'W' (raw_data) not found as sc_high.constant"
        w_result = next(r for r in w_op.results if r.value_id == "W")
        assert w_result.dims == [8, 3, 3, 3], \
            f"Expected [8,3,3,3], got {w_result.dims}"


# ---------------------------------------------------------------------------
# Error-path / edge-case tests
# ---------------------------------------------------------------------------

class TestWeightBufferEdgeCases:

    def test_duplicate_initializer_name_does_not_crash(self, runner, tmp_path):
        """If the ONNX graph somehow contains a duplicate initializer name,
        the driver must exit cleanly (no crash, no assertion failure).
        The WeightBuffer's duplicate-insert guard should handle this."""
        onnx_file = str(tmp_path / "dup.onnx")
        sir_file  = str(tmp_path / "dup.sir")

        W1 = numpy_helper.from_array(np.ones((2, 2),  dtype=np.float32), "W")
        W2 = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), "W")  # dup

        X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
        Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "W"], ["Y"])],
            "dup", [X], [Y], initializer=[W1, W2])

        # Skip onnx.checker here — duplicate initializers are technically invalid
        # ONNX but we want to test our own guard, not the checker.
        model = helper.make_model(graph, opset_imports=_opset())
        with open(onnx_file, "wb") as f:
            f.write(model.SerializeToString())

        result = runner.run(onnx_file, sir_file)
        # Must not segfault or throw — return code 0 or a clean non-zero both acceptable.
        assert result.returncode in (0, 1), \
            f"Driver crashed (returncode={result.returncode}) on duplicate initializer"

    def test_zero_element_initializer(self, runner, tmp_path):
        """A scalar (0-d) weight tensor must not crash the ingressor."""
        onnx_file = str(tmp_path / "scalar_w.onnx")
        sir_file  = str(tmp_path / "scalar_w.sir")

        # ONNX scalar initializer: shape=[] (rank-0)
        scalar = numpy_helper.from_array(
            np.float32(3.14), name="scale")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
        graph = helper.make_graph(
            [helper.make_node("Mul", ["X", "scale"], ["Y"])],
            "scalar-w", [X], [Y], initializer=[scalar])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(onnx_file, "wb") as f:
            f.write(model.SerializeToString())

        result = runner.run(onnx_file, sir_file)
        # Scalar weights may or may not be supported yet — assert no crash.
        assert result.returncode in (0, 1), \
            f"Driver crashed on scalar initializer (returncode={result.returncode})"

    def test_initializer_not_referenced_by_any_node(self, runner, tmp_path):
        """An unreferenced initializer (dead weight) must not crash ingestion.
        It will appear as a constant op with no users — the validator may warn."""
        onnx_file = str(tmp_path / "dead_w.onnx")
        sir_file  = str(tmp_path / "dead_w.sir")

        W_live = numpy_helper.from_array(np.ones((3, 3),  dtype=np.float32), "W_live")
        W_dead = numpy_helper.from_array(np.zeros((3, 3), dtype=np.float32), "W_dead")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "W_live"], ["Y"])],
            "dead-w", [X], [Y], initializer=[W_live, W_dead])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(onnx_file, "wb") as f:
            f.write(model.SerializeToString())

        result = runner.run(onnx_file, sir_file)
        # Dead weights are a warning-level issue, not a hard error.
        assert result.returncode == 0, (
            f"Driver should not fail on unreferenced initializer\n"
            f"[stderr]\n{result.stderr}"
        )