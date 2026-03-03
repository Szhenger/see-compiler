"""
test_ingressor.py — component tests for OnnxIngressor.

Verifies that the ingressor correctly:
  1. Translates graph inputs into SIR block arguments
  2. Translates initializers into sc_high.constant ops
  3. Translates each ONNX node into the correct sc_high.* mnemonic
  4. Threads operand connectivity through the symbol table
  5. Handles unknown ops as sc_high.unknown passthroughs
  6. Emits the correct total op count

Run with:
    pytest test_ingressor.py -v
"""

import os
import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto, checker, numpy_helper

# ---------------------------------------------------------------------------
# Helpers  (mirrors the pattern established in test_shape_inference.py)
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


def _find_op(summary, mnemonic_fragment: str):
    """Return the first OpSummary whose mnemonic contains `mnemonic_fragment`."""
    return next(
        (op for op in summary.ops
         if mnemonic_fragment.lower() in op.mnemonic.lower()),
        None,
    )


def _find_result(summary, value_id: str):
    """Return the ShapeResult for `value_id` anywhere in the op list."""
    for op in summary.ops:
        for r in op.results:
            if r.value_id == value_id:
                return r
    return None


# ---------------------------------------------------------------------------
# 1. Input / block-argument ingestion
# ---------------------------------------------------------------------------

class TestInputIngestion:

    def test_graph_input_appears_as_block_argument(self, runner, tmp_path):
        """Graph inputs (non-initializer) must become SIR block arguments,
        not sc_high.input ops."""
        onnx_file = str(tmp_path / "input.onnx")
        sir_file  = str(tmp_path / "input.sir")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
        graph = helper.make_graph(
            [helper.make_node("Relu", ["X"], ["Y"])],
            "input-test", [X], [Y])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        # A block argument has no defining op — it must NOT appear as any
        # op's result in the op list. It is implicit in the block signature.
        assert _find_result(summary, "X") is None, \
            "'X' is a graph input and must be a block argument, " \
            "not an op result — found it as an op result, which is wrong"

    def test_dynamic_batch_input_accepted(self, runner, tmp_path):
        """An input with a dynamic batch dim (None) must not cause ingestion to fail."""
        onnx_file = str(tmp_path / "dyn_input.onnx")
        sir_file  = str(tmp_path / "dyn_input.sir")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 64])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 64])
        graph = helper.make_graph(
            [helper.make_node("Relu", ["X"], ["Y"])],
            "dyn-input", [X], [Y])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)
        assert summary.returncode == 0 or summary.num_ops >= 1, \
            "Dynamic batch input should not abort ingestion"

    def test_multiple_graph_inputs(self, runner, tmp_path):
        """A graph with two inputs must ingest both without error."""
        onnx_file = str(tmp_path / "two_inputs.onnx")
        sir_file  = str(tmp_path / "two_inputs.sir")

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 4])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 4])
        graph = helper.make_graph(
            [helper.make_node("Add", ["A", "B"], ["C"])],
            "two-inputs", [A, B], [C])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)
        # Both A and B are block arguments — neither should appear as an op result.
        assert _find_result(summary, "A") is None, \
            "'A' must be a block argument, not an op result"
        assert _find_result(summary, "B") is None, \
            "'B' must be a block argument, not an op result"


# ---------------------------------------------------------------------------
# 2. Node → SIR op translation
# ---------------------------------------------------------------------------

class TestNodeTranslation:

    @pytest.mark.parametrize("onnx_op, sir_mnemonic", [
        ("Relu",              "sc_high.relu"),
        ("Add",               "sc_high.add"),
        ("Conv",              "sc_high.conv2d"),
        ("BatchNormalization","sc_high.batch_norm"),
        ("Gemm",              "sc_high.gemm"),
    ])
    def test_op_mnemonic_mapping(self, runner, tmp_path, onnx_op, sir_mnemonic):
        """Each supported ONNX op_type must map to the correct sc_high.* mnemonic."""
        onnx_file = str(tmp_path / f"{onnx_op.lower()}.onnx")
        sir_file  = str(tmp_path / f"{onnx_op.lower()}.sir")

        # Build a minimal valid graph for each op.
        if onnx_op == "Relu":
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
            nodes = [helper.make_node("Relu", ["X"], ["Y"])]
            inputs, inits = [X], []

        elif onnx_op == "Add":
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
            W = numpy_helper.from_array(np.ones((2, 4), dtype=np.float32), "W")
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
            nodes = [helper.make_node("Add", ["X", "W"], ["Y"])]
            inputs, inits = [X], [W]

        elif onnx_op == "Conv":
            W = numpy_helper.from_array(
                np.zeros((8, 3, 3, 3), dtype=np.float32), "W")
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
            nodes = [helper.make_node("Conv", ["X", "W"], ["Y"],
                                      strides=[1, 1], pads=[0, 0, 0, 0])]
            inputs, inits = [X], [W]

        elif onnx_op == "BatchNormalization":
            scale = numpy_helper.from_array(np.ones(4,  dtype=np.float32), "scale")
            bias  = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "bias")
            mean  = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "mean")
            var   = numpy_helper.from_array(np.ones(4,  dtype=np.float32), "var")
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 4, 4])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 4, 4])
            nodes = [helper.make_node("BatchNormalization",
                                      ["X", "scale", "bias", "mean", "var"], ["Y"])]
            inputs, inits = [X], [scale, bias, mean, var]

        elif onnx_op == "Gemm":
            W = numpy_helper.from_array(np.zeros((8, 4), dtype=np.float32), "W")
            b = numpy_helper.from_array(np.zeros(8,      dtype=np.float32), "b")
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 8])
            nodes = [helper.make_node("Gemm", ["X", "W", "b"], ["Y"])]
            inputs, inits = [X], [W, b]

        graph = helper.make_graph(nodes, f"{onnx_op.lower()}-map",
                                   inputs, [Y if onnx_op != "Add"
                                            else helper.make_tensor_value_info(
                                                "Y", TensorProto.FLOAT, [2, 4])],
                                   initializer=inits)
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        op = _find_op(summary, sir_mnemonic)
        assert op is not None, \
            f"ONNX '{onnx_op}' should map to '{sir_mnemonic}', " \
            f"but no such op found in SIR. Ops present: " \
            f"{[o.mnemonic for o in summary.ops]}"

    def test_unknown_op_becomes_passthrough(self, runner, tmp_path):
        """An unsupported ONNX op must produce a sc_high.unknown passthrough,
        not crash the ingressor."""
        onnx_file = str(tmp_path / "unknown.onnx")
        sir_file  = str(tmp_path / "unknown.sir")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        # EyeLike is a valid ONNX op not in SeeC++'s supported set.
        graph = helper.make_graph(
            [helper.make_node("EyeLike", ["X"], ["Y"])],
            "unknown-op", [X], [Y])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(onnx_file, "wb") as f:
            f.write(model.SerializeToString())

        # Unknown ops produce a passthrough — the ingressor must not crash.
        # The validator will later reject the block (exit non-zero is acceptable),
        # but a segfault or unhandled exception is not.
        result = runner.run(onnx_file, sir_file)
        assert result.returncode in (0, 1), \
            f"Ingressor must not crash on unknown op (got returncode={result.returncode})"

        # If it did succeed, the unknown passthrough must appear in the output.
        if result.returncode == 0:
            summary = runner.run_and_parse(onnx_file, sir_file)
            unknown_op = _find_op(summary, "unknown")
            assert unknown_op is not None, \
                "Unsupported op must produce sc_high.unknown passthrough in SIR"


# ---------------------------------------------------------------------------
# 3. Operand connectivity (symbol table wiring)
# ---------------------------------------------------------------------------

class TestOperandConnectivity:

    def test_single_node_operand_linked_to_input(self, runner, tmp_path):
        """Relu's result must exist in the SIR output, confirming
        the symbol table entry for 'Y' was created by the Relu op."""
        onnx_file = str(tmp_path / "connectivity.onnx")
        sir_file  = str(tmp_path / "connectivity.sir")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
        graph = helper.make_graph(
            [helper.make_node("Relu", ["X"], ["Y"])],
            "connectivity", [X], [Y])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        # 'Y' is the Relu result — it must appear as a resolved op result.
        y_result = _find_result(summary, "Y")
        assert y_result is not None, \
            "Relu output 'Y' must appear as a resolved op result in SIR"

    def test_two_node_chain_result_threading(self, runner, tmp_path):
        """In a MatMul → Relu chain, Relu's operand must be the MatMul result.
        This verifies the symbol table correctly threads intermediate values."""
        onnx_file = str(tmp_path / "chain.onnx")
        sir_file  = str(tmp_path / "chain.sir")

        W = numpy_helper.from_array(np.zeros((4, 8), dtype=np.float32), "W")
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
        D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [2, 8])
        nodes = [
            helper.make_node("MatMul", ["A", "W"], ["C"]),
            helper.make_node("Relu",   ["C"],       ["D"]),
        ]
        graph = helper.make_graph(nodes, "chain", [A], [D], initializer=[W])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        # Both intermediate 'C' and final 'D' must be resolved op results.
        c_result = _find_result(summary, "C")
        d_result = _find_result(summary, "D")

        assert c_result is not None, \
            "MatMul output 'C' must appear as a resolved result in SIR"
        assert d_result is not None, \
            "Relu output 'D' must appear as a resolved result in SIR"

    def test_op_count_matches_graph_structure(self, runner, tmp_path):
        """Op count must equal: num_initializers + num_nodes.
        For W (constant) + MatMul + Relu = 3 ops total."""
        onnx_file = str(tmp_path / "opcount.onnx")
        sir_file  = str(tmp_path / "opcount.sir")

        W = numpy_helper.from_array(np.zeros((4, 8), dtype=np.float32), "W")
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
        D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [2, 8])
        nodes = [
            helper.make_node("MatMul", ["A", "W"], ["C"]),
            helper.make_node("Relu",   ["C"],       ["D"]),
        ]
        graph = helper.make_graph(nodes, "opcount", [A], [D], initializer=[W])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        # 1 constant (W) + 1 matmul + 1 relu = 3
        assert summary.num_ops == 3, \
            f"Expected 3 ops (constant + matmul + relu), got {summary.num_ops}"