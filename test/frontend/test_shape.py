"""
test_shape_inference.py — component-level tests for ShapeInferencePass.

Each test class covers one op handler. The parametrize dimensions are chosen
to exercise:
  - typical production shapes (ResNet, BERT)
  - edge cases (rank-2 only, batch=1, large channel counts)
  - dynamic batch dimensions (-1 / kDynamic)

Run with:
    pytest test_shape_inference.py -v
"""

import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto, checker, numpy_helper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_model(path, model: onnx.ModelProto) -> None:
    """Validate and serialise an ONNX model to disk."""
    try:
        checker.check_model(model)
    except checker.ValidationError as e:
        # Surface fixture problems as test errors, not compiler failures.
        raise ValueError(f"Test fixture is not valid ONNX: {e}") from e
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def _opset(version: int = 17):
    return [helper.make_opsetid("", version)]


# ---------------------------------------------------------------------------
# MatMul shape inference
# ---------------------------------------------------------------------------

class TestMatMulShape:

    @pytest.mark.parametrize("m, k, n", [
        (2,   4,   8),      # small
        (1,  128,  64),     # BERT-style sequence projection
        (32,  512, 512),    # attention head
    ])
    def test_2d_output_shape(self, runner, tmp_path, m, k, n):
        """[M,K] @ [K,N] -> [M,N]"""
        onnx_file = str(tmp_path / "matmul.onnx")
        sir_file  = str(tmp_path / "matmul.sir")

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [m, n])
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["A", "B"], ["C"])],
            "matmul", [A, B], [C])
        _write_model(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        # Find the resolved result for C in the logger output.
        result = next(
            (r for op in summary.ops for r in op.results if r.value_id == "C"),
            None,
        )
        assert result is not None, \
            "Shape inference did not emit a resolved shape for 'C'"
        assert result.dims == [m, n], \
            f"Expected [{m},{n}], got {result.dims}"

    @pytest.mark.parametrize("k_mismatch", [3, 9, 100])
    def test_inner_dim_mismatch_fails(self, runner, tmp_path, k_mismatch):
        """[2,4] @ [K_wrong, 8] must cause the driver to exit non-zero."""
        onnx_file = str(tmp_path / "bad_matmul.onnx")
        sir_file  = str(tmp_path / "bad_matmul.sir")

        # Intentionally build an invalid graph — skip checker.
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k_mismatch, 8])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 8])
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["A", "B"], ["C"])],
            "bad-matmul", [A, B], [C])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(onnx_file, "wb") as f:
            f.write(model.SerializeToString())

        result = runner.run(onnx_file, sir_file)
        assert result.returncode != 0, \
            "Driver should fail on inner dimension mismatch"


# ---------------------------------------------------------------------------
# Conv2D shape inference
# ---------------------------------------------------------------------------

class TestConv2DShape:

    def _make_conv_model(self, input_shape, filter_shape,
                         strides=(1, 1), pads=(0, 0, 0, 0),
                         dilations=(1, 1), group=1):
        """Build a minimal Conv model with a weight initializer."""
        W_data = np.zeros(filter_shape, dtype=np.float32)
        W_init = numpy_helper.from_array(W_data, name="W")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_shape))
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)   # inferred

        node = helper.make_node(
            "Conv", ["X", "W"], ["Y"],
            strides=list(strides),
            pads=list(pads),
            dilations=list(dilations),
            group=group,
        )
        graph = helper.make_graph([node], "conv", [X], [Y],
                                   initializer=[W_init])
        return helper.make_model(graph, opset_imports=_opset())

    @pytest.mark.parametrize("N,C,H,W,F,KH,KW,expected_h,expected_w", [
        # no padding, stride 1:  out = H - K + 1
        (1, 3, 8,  8,  8, 3, 3, 6,  6 ),
        (1, 3, 32, 32, 16, 3, 3, 30, 30),
        # same padding (pad=1), stride 1: out = H
        (1, 3, 8,  8,  8, 3, 3, 8,  8 ),   # pads=[1,1,1,1]
    ])
    def test_output_spatial_dims(
        self, runner, tmp_path,
        N, C, H, W, F, KH, KW, expected_h, expected_w
    ):
        """Conv output spatial dims follow the ONNX formula."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")

        # Third parametrized case uses same-padding.
        pads = (1, 1, 1, 1) if (expected_h == H) else (0, 0, 0, 0)
        model = self._make_conv_model(
            input_shape=(N, C, H, W),
            filter_shape=(F, C, KH, KW),
            pads=pads,
        )
        _write_model(onnx_file, model)

        summary = runner.run_and_parse(onnx_file, sir_file)
        result = next(
            (r for op in summary.ops for r in op.results if r.value_id == "Y"),
            None,
        )
        assert result is not None, "No resolved shape for Conv output 'Y'"
        assert result.dims == [N, F, expected_h, expected_w], \
            f"Expected [{N},{F},{expected_h},{expected_w}], got {result.dims}"

    def test_strided_conv_halves_spatial(self, runner, tmp_path):
        """stride=2 on a 16x16 input with no padding -> 8x8 output."""
        onnx_file = str(tmp_path / "strided_conv.onnx")
        sir_file  = str(tmp_path / "strided_conv.sir")

        model = self._make_conv_model(
            input_shape=(1, 4, 16, 16),
            filter_shape=(8, 4, 3, 3),
            strides=(2, 2),
            pads=(1, 1, 1, 1),   # same-padding with stride 2 -> output = input/2
        )
        _write_model(onnx_file, model)

        summary = runner.run_and_parse(onnx_file, sir_file)
        result = next(
            (r for op in summary.ops for r in op.results if r.value_id == "Y"),
            None,
        )
        assert result is not None
        assert result.dims == [1, 8, 8, 8], \
            f"Expected [1,8,8,8] for strided conv, got {result.dims}"

    def test_dilated_conv_output_shape(self, runner, tmp_path):
        """Dilation=2 effectively expands kernel size: effective_k = d*(k-1)+1."""
        # input 8x8, kernel 3x3, dilation 2, no pad:
        # effective kernel = 2*(3-1)+1 = 5 -> out = (8-5)/1 + 1 = 4
        onnx_file = str(tmp_path / "dilated_conv.onnx")
        sir_file  = str(tmp_path / "dilated_conv.sir")

        model = self._make_conv_model(
            input_shape=(1, 1, 8, 8),
            filter_shape=(4, 1, 3, 3),
            dilations=(2, 2),
        )
        _write_model(onnx_file, model)

        summary = runner.run_and_parse(onnx_file, sir_file)
        result = next(
            (r for op in summary.ops for r in op.results if r.value_id == "Y"),
            None,
        )
        assert result is not None
        assert result.dims == [1, 4, 4, 4], \
            f"Expected [1,4,4,4] for dilated conv, got {result.dims}"


# ---------------------------------------------------------------------------
# Elementwise / broadcast shape inference
# ---------------------------------------------------------------------------

class TestElementwiseShape:

    @pytest.mark.parametrize("shape", [
        [2, 8],
        [1, 3, 224, 224],
        [32, 128],
    ])
    def test_relu_preserves_shape(self, runner, tmp_path, shape):
        """Relu output must be identical in shape and dtype to its input."""
        onnx_file = str(tmp_path / "relu.onnx")
        sir_file  = str(tmp_path / "relu.sir")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)
        graph = helper.make_graph(
            [helper.make_node("Relu", ["X"], ["Y"])],
            "relu", [X], [Y])
        _write_model(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)
        result  = next(
            (r for op in summary.ops for r in op.results if r.value_id == "Y"),
            None,
        )
        assert result is not None, "No resolved shape for Relu output 'Y'"
        assert result.dims == shape, \
            f"Relu shape mismatch: expected {shape}, got {result.dims}"

    def test_add_broadcast_shape(self, runner, tmp_path):
        """Add([1,8], [4,8]) -> [4,8] via NumPy broadcasting."""
        onnx_file = str(tmp_path / "add_broadcast.onnx")
        sir_file  = str(tmp_path / "add_broadcast.sir")

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 8])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 8])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4, 8])
        graph = helper.make_graph(
            [helper.make_node("Add", ["A", "B"], ["C"])],
            "add-bc", [A, B], [C])
        _write_model(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)
        result  = next(
            (r for op in summary.ops for r in op.results if r.value_id == "C"),
            None,
        )
        assert result is not None
        assert result.dims == [4, 8], \
            f"Broadcast Add shape: expected [4,8], got {result.dims}"

    def test_add_incompatible_shapes_fails(self, runner, tmp_path):
        """Add([2,4], [3,4]) must cause driver to exit non-zero."""
        onnx_file = str(tmp_path / "bad_add.onnx")
        sir_file  = str(tmp_path / "bad_add.sir")

        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 4])
        graph = helper.make_graph(
            [helper.make_node("Add", ["A", "B"], ["C"])],
            "bad-add", [A, B], [C])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(onnx_file, "wb") as f:
            f.write(model.SerializeToString())

        result = runner.run(onnx_file, sir_file)
        assert result.returncode != 0, \
            "Driver should fail on broadcast-incompatible Add shapes"


# ---------------------------------------------------------------------------
# BatchNorm shape inference
# ---------------------------------------------------------------------------

class TestBatchNormShape:

    @pytest.mark.parametrize("N,C,H,W", [
        (1,  64, 56, 56),   # ResNet layer 1
        (4, 256,  7,  7),   # ResNet layer 4
    ])
    def test_output_matches_input(self, runner, tmp_path, N, C, H, W):
        """BatchNorm is shape-preserving: output dims == input dims."""
        onnx_file = str(tmp_path / "bn.onnx")
        sir_file  = str(tmp_path / "bn.sir")

        scale = numpy_helper.from_array(np.ones(C,  dtype=np.float32), "scale")
        bias  = numpy_helper.from_array(np.zeros(C, dtype=np.float32), "bias")
        mean  = numpy_helper.from_array(np.zeros(C, dtype=np.float32), "mean")
        var   = numpy_helper.from_array(np.ones(C,  dtype=np.float32), "var")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, C, H, W])

        node  = helper.make_node(
            "BatchNormalization",
            ["X", "scale", "bias", "mean", "var"], ["Y"])
        graph = helper.make_graph([node], "bn", [X], [Y],
                                   initializer=[scale, bias, mean, var])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)
        result  = next(
            (r for op in summary.ops for r in op.results if r.value_id == "Y"),
            None,
        )
        assert result is not None, "No resolved shape for BatchNorm output 'Y'"
        assert result.dims == [N, C, H, W], \
            f"BatchNorm shape mismatch: expected {[N,C,H,W]}, got {result.dims}"


# ---------------------------------------------------------------------------
# Dynamic batch dimension propagation
# ---------------------------------------------------------------------------

class TestDynamicShapes:

    def test_dynamic_batch_propagates_through_relu(self, runner, tmp_path):
        """A -1 (dynamic) batch dim must propagate to Relu output, not cause failure."""
        onnx_file = str(tmp_path / "dyn_relu.onnx")
        sir_file  = str(tmp_path / "dyn_relu.sir")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 64])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 64])
        graph = helper.make_graph(
            [helper.make_node("Relu", ["X"], ["Y"])],
            "dyn-relu", [X], [Y])
        _write_model(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        # Dynamic shapes must not crash the pipeline.
        summary = runner.run_and_parse(onnx_file, sir_file)
        result  = next(
            (r for op in summary.ops for r in op.results if r.value_id == "Y"),
            None,
        )
        assert result is not None
        # Dynamic dim is represented as -1 in our SIR (kDynamic).
        assert result.dims[0] == -1, \
            f"Expected dynamic dim -1 at index 0, got {result.dims[0]}"
        assert result.dims[1] == 64