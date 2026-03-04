"""
test_passes.py — isolated stress tests for each middle-end pass.

Where test_middle_end.py tests the full 5-pass pipeline and asserts on
aggregate Logger output, this file tests each pass's internal contracts
in isolation — the shape arithmetic, guard conditions, erasure ordering,
and edge cases that are invisible at pipeline level.

Coverage:
  TestConvLoweringShapeArithmetic  — im2col/matmul output dims are exactly right
  TestConvLoweringGuards           — non-conv untouched, chained conv, empty block
  TestFusionGuards                 — hasOneUse, isConstant, non-conv BN input
  TestFusionPatterns               — all three patterns, N-pair scaling, post-lowering
  TestDCELiveness                  — seed correctness, propagation, erasure ordering
  TestDCEEdgeCases                 — mostly-dead block, side-effecting op guard
  TestPassInteraction              — fusion-then-DCE composition, full pass sequence

Logger contract (lines parsed in this file):
  ConvLoweringPass: lowering N conv2d op(s)
  ConvLoweringPass: conv2d -> im2col + reshape + matmul[+ add(bias)]
  OperatorFusionPass: fused N conv+bn, N matmul+relu, N elementwise chain(s)
  DeadCodeEliminationPass: removed N dead op(s)

Run with:
    pytest test_passes.py -v
"""

import re
import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto, checker, numpy_helper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opset(v: int = 17):
    return [helper.make_opsetid("", v)]


def _write(path: str, model: onnx.ModelProto, skip_check: bool = False) -> None:
    if not skip_check:
        try:
            checker.check_model(model)
        except checker.ValidationError as e:
            raise ValueError(f"Bad fixture: {e}") from e
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def _combined(summary) -> str:
    return summary.stderr + summary.stdout


def _lowering_count(summary) -> int:
    m = re.search(r"ConvLoweringPass: lowering\s+(\d+)", _combined(summary))
    return int(m.group(1)) if m else 0


def _fusion_counts(summary) -> dict:
    m = re.search(
        r"OperatorFusionPass: fused\s+(\d+)\s+conv\+bn,"
        r"\s+(\d+)\s+matmul\+relu,\s+(\d+)\s+elementwise",
        _combined(summary),
    )
    if not m:
        return {"conv_bn": 0, "matmul_relu": 0, "elementwise": 0}
    return {
        "conv_bn":     int(m.group(1)),
        "matmul_relu": int(m.group(2)),
        "elementwise": int(m.group(3)),
    }


def _dce_counts(summary) -> list:
    """List of removal counts, one entry per DCE pass execution."""
    return [int(m.group(1))
            for m in re.finditer(
                r"DeadCodeEliminationPass: removed\s+(\d+)", summary.stderr)]


def _total_dce(summary) -> int:
    return sum(_dce_counts(summary))


# ---------------------------------------------------------------------------
# Conv output dim formula — mirrors ConvLoweringPass::conv_dim exactly
# ---------------------------------------------------------------------------

def _conv_out_dim(in_: int, k: int, s: int, pb: int, pe: int, d: int) -> int:
    """floor((in + pb + pe - d*(k-1) - 1) / s + 1)"""
    return (in_ + pb + pe - d * (k - 1) - 1) // s + 1


# ===========================================================================
# 1. ConvLoweringPass — shape arithmetic
#
# For each fixture we compute expected shapes from first principles and
# assert the Logger contains a matching dimension string.
#
#   im2col  : [N,  C*KH*KW,  out_H*out_W]
#   reshape : [F,  C*KH*KW]
#   matmul  : [N,  F,  out_H*out_W]
# ===========================================================================

class TestConvLoweringShapeArithmetic:

    @pytest.mark.parametrize("N,C,H,W,F,KH,KW,s,p", [
        (1,  3,  8,  8,  8, 3, 3, 1, 0),   # no-pad stride-1  -> out=6x6
        (1,  3,  8,  8,  8, 3, 3, 1, 1),   # same-pad stride-1 -> out=8x8
        (1,  3, 16, 16, 16, 3, 3, 2, 1),   # same-pad stride-2 -> out=8x8
        (2,  8, 14, 14, 32, 1, 1, 1, 0),   # pointwise 1x1     -> out=14x14
        (1, 32,  7,  7, 64, 3, 3, 1, 1),   # ResNet layer-4 block
    ])
    def test_im2col_output_shape_logged(
        self, runner, tmp_path, N, C, H, W, F, KH, KW, s, p
    ):
        """im2col result shape [N, C*KH*KW, out_H*out_W] must match the formula."""
        onnx_file = str(tmp_path / "conv_im2col.onnx")
        sir_file  = str(tmp_path / "conv_im2col.sir")

        W_init = numpy_helper.from_array(
            np.zeros((F, C, KH, KW), dtype=np.float32), "W")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        node = helper.make_node("Conv", ["X", "W"], ["Y"],
                                 strides=[s, s], pads=[p, p, p, p])
        graph = helper.make_graph([node], "im2col-shape", [X], [Y],
                                   initializer=[W_init])
        _write(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        out_H    = _conv_out_dim(H, KH, s, p, p, 1)
        out_W    = _conv_out_dim(W, KW, s, p, p, 1)
        col_rows = C * KH * KW
        col_cols = out_H * out_W
        expected = f"{N}x{col_rows}x{col_cols}"

        assert expected in _combined(summary), \
            f"Expected im2col shape [{N},{col_rows},{col_cols}] in Logger; " \
            f"params=(N={N},C={C},H={H},W={W},F={F},K={KH},s={s},p={p})"

    @pytest.mark.parametrize("N,C,H,W,F,KH,KW,s,p", [
        (1,  3,  8,  8,  8, 3, 3, 1, 0),
        (1,  3,  8,  8,  8, 3, 3, 1, 1),
        (1, 32,  7,  7, 64, 3, 3, 1, 1),
    ])
    def test_matmul_output_shape_logged(
        self, runner, tmp_path, N, C, H, W, F, KH, KW, s, p
    ):
        """sc_low.matmul result shape [N, F, out_H*out_W] must be logged."""
        onnx_file = str(tmp_path / "conv_mm.onnx")
        sir_file  = str(tmp_path / "conv_mm.sir")

        W_init = numpy_helper.from_array(
            np.zeros((F, C, KH, KW), dtype=np.float32), "W")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        node = helper.make_node("Conv", ["X", "W"], ["Y"],
                                 strides=[s, s], pads=[p, p, p, p])
        graph = helper.make_graph([node], "mm-shape", [X], [Y],
                                   initializer=[W_init])
        _write(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        out_H    = _conv_out_dim(H, KH, s, p, p, 1)
        out_W    = _conv_out_dim(W, KW, s, p, p, 1)
        col_cols = out_H * out_W
        expected = f"{N}x{F}x{col_cols}"

        assert expected in _combined(summary), \
            f"Expected matmul output [{N},{F},{col_cols}] in Logger; " \
            f"params=(N={N},C={C},H={H},W={W},F={F},K={KH},s={s},p={p})"

    @pytest.mark.parametrize("dilation,expected_out", [
        (1, 6),   # (8 + 0 - 1*(3-1) - 1) / 1 + 1 = 6
        (2, 4),   # (8 + 0 - 2*(3-1) - 1) / 1 + 1 = 4
        (3, 2),   # (8 + 0 - 3*(3-1) - 1) / 1 + 1 = 2
    ])
    def test_dilated_conv_im2col_shape(
        self, runner, tmp_path, dilation, expected_out
    ):
        """Dilation expands effective kernel size. im2col col_cols must
        use the ONNX dilation formula, not a naive k-based formula."""
        onnx_file = str(tmp_path / f"dil{dilation}.onnx")
        sir_file  = str(tmp_path / f"dil{dilation}.sir")

        N, C, H, W, F, KH, KW = 1, 4, 8, 8, 8, 3, 3
        W_init = numpy_helper.from_array(
            np.zeros((F, C, KH, KW), dtype=np.float32), "W")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        node = helper.make_node("Conv", ["X", "W"], ["Y"],
                                 strides=[1, 1], pads=[0, 0, 0, 0],
                                 dilations=[dilation, dilation])
        graph = helper.make_graph([node], "dil-conv", [X], [Y],
                                   initializer=[W_init])
        _write(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        col_rows = C * KH * KW
        col_cols = expected_out * expected_out
        expected = f"{N}x{col_rows}x{col_cols}"

        assert expected in _combined(summary), \
            f"Dilated conv (d={dilation}): expected im2col " \
            f"[{N},{col_rows},{col_cols}]; out_H={expected_out}"

    def test_filter_reshape_dims_logged(self, runner, tmp_path):
        """sc_low.reshape must produce [F, C*KH*KW] — the filter row matrix.
        This verifies the reshape step is emitted with the correct dimensions,
        not just that lowering ran."""
        N, C, H, W, F, KH, KW = 1, 3, 8, 8, 16, 3, 3
        onnx_file = str(tmp_path / "reshape.onnx")
        sir_file  = str(tmp_path / "reshape.sir")

        W_init = numpy_helper.from_array(
            np.zeros((F, C, KH, KW), dtype=np.float32), "W")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        node = helper.make_node("Conv", ["X", "W"], ["Y"],
                                 strides=[1,1], pads=[0,0,0,0])
        graph = helper.make_graph([node], "reshape-test", [X], [Y],
                                   initializer=[W_init])
        _write(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        col_rows = C * KH * KW   # 3*3*3 = 27
        expected = f"{F}x{col_rows}"   # "16x27"

        assert expected in _combined(summary), \
            f"Expected filter reshape dims [{F},{col_rows}] in Logger output"

    def test_bias_add_shape_matches_matmul(self, runner, tmp_path):
        """sc_low.add (bias) must carry the same shape as sc_low.matmul output.
        The shape string must appear at least twice: once for matmul, once for add."""
        N, C, H, W, F, KH, KW = 1, 3, 8, 8, 8, 3, 3
        onnx_file = str(tmp_path / "bias_shape.onnx")
        sir_file  = str(tmp_path / "bias_shape.sir")

        W_init = numpy_helper.from_array(
            np.zeros((F, C, KH, KW), dtype=np.float32), "W")
        b_init = numpy_helper.from_array(np.zeros(F, dtype=np.float32), "b")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        node = helper.make_node("Conv", ["X", "W", "b"], ["Y"],
                                 strides=[1,1], pads=[0,0,0,0])
        graph = helper.make_graph([node], "bias-shape", [X], [Y],
                                   initializer=[W_init, b_init])
        _write(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        out_H    = _conv_out_dim(H, KH, 1, 0, 0, 1)
        out_W    = _conv_out_dim(W, KW, 1, 0, 0, 1)
        col_cols = out_H * out_W
        expected = f"{N}x{F}x{col_cols}"

        occurrences = _combined(summary).count(expected)
        assert occurrences >= 2, \
            f"Shape {expected} must appear >= 2 times (matmul + bias add); " \
            f"got {occurrences}"


# ===========================================================================
# 2. ConvLoweringPass — guard conditions
# ===========================================================================

class TestConvLoweringGuards:

    def test_non_conv_ops_untouched(self, runner, tmp_path):
        """A block with only MatMul must report 0 conv lowerings."""
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 16])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4, 16])
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["A", "B"], ["C"])],
            "no-conv", [A, B], [C])
        _write(str(tmp_path / "no_conv.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "no_conv.onnx"),
            str(tmp_path / "no_conv.sir"))

        assert _lowering_count(summary) == 0, \
            "MatMul-only block: lowering count must be 0"

    def test_conv_chain_both_lowered(self, runner, tmp_path):
        """Two chained Conv ops must both be lowered; the block must remain
        structurally valid after both rewrites."""
        W1 = numpy_helper.from_array(
            np.zeros((8, 3, 3, 3), dtype=np.float32), "W1")
        W2 = numpy_helper.from_array(
            np.zeros((16, 8, 1, 1), dtype=np.float32), "W2")
        X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [1, 3, 8, 8])
        Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, None)
        nodes = [
            helper.make_node("Conv", ["X",  "W1"], ["c1"],
                              strides=[1,1], pads=[0,0,0,0]),
            helper.make_node("Conv", ["c1", "W2"], ["Y"],
                              strides=[1,1], pads=[0,0,0,0]),
        ]
        graph = helper.make_graph(nodes, "conv-chain", [X], [Y],
                                   initializer=[W1, W2])
        _write(str(tmp_path / "chain.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "chain.onnx"),
            str(tmp_path / "chain.sir"))

        assert _lowering_count(summary) == 2, \
            "Chained two-conv block must lower both ops"
        assert summary.returncode == 0, \
            "Block must remain valid after chained conv lowering"

    def test_gemm_only_block_lowering_count_zero(self, runner, tmp_path):
        """A Gemm-only block must run ConvLoweringPass with 0 lowerings —
        the pass must handle the no-Conv case without crashing."""
        W = numpy_helper.from_array(np.zeros((4,8), dtype=np.float32), "W")
        b = numpy_helper.from_array(np.zeros(8,     dtype=np.float32), "b")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8])
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["X","W","b"], ["Y"])],
            "gemm-only", [X], [Y], initializer=[W, b])
        _write(str(tmp_path / "gemm.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "gemm.onnx"),
            str(tmp_path / "gemm.sir"))

        assert _lowering_count(summary) == 0
        assert summary.returncode == 0


# ===========================================================================
# 3. OperatorFusionPass — guard conditions (negative cases)
# ===========================================================================

class TestFusionGuards:

    def test_relu_with_two_users_not_fused(self, runner, tmp_path):
        """If Relu feeds two downstream ops, hasOneUse() is false and the
        fusion must NOT fire — fusing would change the value seen by the
        second user."""
        W  = numpy_helper.from_array(np.zeros((4,8), dtype=np.float32), "W")
        b  = numpy_helper.from_array(np.zeros(8,     dtype=np.float32), "b")
        W2 = numpy_helper.from_array(np.zeros((8,8), dtype=np.float32), "W2")
        b2 = numpy_helper.from_array(np.zeros(8,     dtype=np.float32), "b2")
        W3 = numpy_helper.from_array(np.zeros((8,8), dtype=np.float32), "W3")
        b3 = numpy_helper.from_array(np.zeros(8,     dtype=np.float32), "b3")

        X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [4, 4])
        Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [4, 8])
        Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [4, 8])

        # relu_out feeds BOTH Gemm2 and Gemm3.
        nodes = [
            helper.make_node("Gemm", ["X",         "W",  "b"],  ["g1"]),
            helper.make_node("Relu", ["g1"],                     ["relu_out"]),
            helper.make_node("Gemm", ["relu_out",  "W2", "b2"], ["Y1"]),
            helper.make_node("Gemm", ["relu_out",  "W3", "b3"], ["Y2"]),
        ]
        graph = helper.make_graph(nodes, "shared-relu", [X], [Y1, Y2],
                                   initializer=[W, b, W2, b2, W3, b3])
        _write(str(tmp_path / "shared_relu.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "shared_relu.onnx"),
            str(tmp_path / "shared_relu.sir"))

        counts = _fusion_counts(summary)
        assert counts["matmul_relu"] == 0, \
            f"Relu with 2 users must NOT fuse; got matmul_relu={counts['matmul_relu']}"
        assert summary.returncode == 0, \
            "Block must stay valid when hasOneUse() guard fires"

    def test_bn_after_non_conv_not_fused(self, runner, tmp_path):
        """BatchNorm whose input is a graph argument (not a Conv result)
        must NOT fuse — the definingOp mnemonic check must reject it."""
        scale = numpy_helper.from_array(np.ones(8,  dtype=np.float32), "scale")
        bias  = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "bias")
        mean  = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "mean")
        var   = numpy_helper.from_array(np.ones(8,  dtype=np.float32), "var")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 4, 4])
        node = helper.make_node("BatchNormalization",
                                 ["X","scale","bias","mean","var"], ["Y"])
        graph = helper.make_graph([node], "bn-only", [X], [Y],
                                   initializer=[scale, bias, mean, var])
        _write(str(tmp_path / "bn_only.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "bn_only.onnx"),
            str(tmp_path / "bn_only.sir"))

        counts = _fusion_counts(summary)
        assert counts["conv_bn"] == 0, \
            f"BN after non-Conv must NOT fuse; got conv_bn={counts['conv_bn']}"
        assert summary.returncode == 0

    def test_conv_with_shared_output_not_fused(self, runner, tmp_path):
        """Conv whose output feeds both BN and a second Conv:
        hasOneUse() is false — Conv+BN fusion must NOT fire."""
        W     = numpy_helper.from_array(
            np.zeros((4, 3, 3, 3), dtype=np.float32), "W")
        scale = numpy_helper.from_array(np.ones(4,  dtype=np.float32), "scale")
        bias  = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "bias")
        mean  = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "mean")
        var   = numpy_helper.from_array(np.ones(4,  dtype=np.float32), "var")
        W2    = numpy_helper.from_array(
            np.zeros((4, 4, 1, 1), dtype=np.float32), "W2")

        X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [1, 3, 6, 6])
        Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, None)
        Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, None)

        nodes = [
            helper.make_node("Conv", ["X","W"], ["conv_out"],
                              strides=[1,1], pads=[0,0,0,0]),
            helper.make_node("BatchNormalization",
                              ["conv_out","scale","bias","mean","var"], ["Y1"]),
            helper.make_node("Conv", ["conv_out","W2"], ["Y2"],
                              strides=[1,1], pads=[0,0,0,0]),
        ]
        graph = helper.make_graph(nodes, "shared-conv", [X], [Y1, Y2],
                                   initializer=[W, scale, bias, mean, var, W2])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(str(tmp_path / "shared.onnx"), "wb") as f:
            f.write(model.SerializeToString())

        summary = runner.run_and_parse(
            str(tmp_path / "shared.onnx"),
            str(tmp_path / "shared.sir"))

        counts = _fusion_counts(summary)
        assert counts["conv_bn"] == 0, \
            f"Conv with multiple users must NOT fuse with BN; " \
            f"got conv_bn={counts['conv_bn']}"

    def test_conv_bn_with_dynamic_stats_not_fused(self, runner, tmp_path):
        """BN whose scale/bias/mean/var are graph inputs (not constants)
        must NOT fuse — the isConstant guard must reject them."""
        W = numpy_helper.from_array(
            np.zeros((4, 3, 3, 3), dtype=np.float32), "W")

        X     = helper.make_tensor_value_info("X",     TensorProto.FLOAT, [1,3,6,6])
        scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4])
        bias  = helper.make_tensor_value_info("bias",  TensorProto.FLOAT, [4])
        mean  = helper.make_tensor_value_info("mean",  TensorProto.FLOAT, [4])
        var   = helper.make_tensor_value_info("var",   TensorProto.FLOAT, [4])
        Y     = helper.make_tensor_value_info("Y",     TensorProto.FLOAT, None)

        nodes = [
            helper.make_node("Conv", ["X","W"], ["conv_out"],
                              strides=[1,1], pads=[0,0,0,0]),
            helper.make_node("BatchNormalization",
                              ["conv_out","scale","bias","mean","var"], ["Y"]),
        ]
        # BN params are graph inputs, NOT initializers — isConstant returns false.
        graph = helper.make_graph(
            nodes, "dyn-bn", [X, scale, bias, mean, var], [Y],
            initializer=[W])
        _write(str(tmp_path / "dyn_bn.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "dyn_bn.onnx"),
            str(tmp_path / "dyn_bn.sir"))

        counts = _fusion_counts(summary)
        assert counts["conv_bn"] == 0, \
            f"Dynamic BN stats must block Conv+BN fusion; got conv_bn={counts['conv_bn']}"


# ===========================================================================
# 4. OperatorFusionPass — positive pattern coverage
# ===========================================================================

class TestFusionPatterns:

    def test_mul_add_elementwise_chain_fuses(self, runner, tmp_path):
        """Mul -> Add (both elementwise, single-use intermediate) must trigger
        elementwise chain fusion into sc_high.fused_ew."""
        W1 = numpy_helper.from_array(np.ones((4,4), dtype=np.float32), "W1")
        W2 = numpy_helper.from_array(np.ones((4,4), dtype=np.float32), "W2")
        X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4])
        Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])
        nodes = [
            helper.make_node("Mul", ["X",       "W1"], ["mul_out"]),
            helper.make_node("Add", ["mul_out", "W2"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "mul-add", [X], [Y],
                                   initializer=[W1, W2])
        _write(str(tmp_path / "mul_add.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "mul_add.onnx"),
            str(tmp_path / "mul_add.sir"))

        counts = _fusion_counts(summary)
        assert counts["elementwise"] >= 1, \
            f"Mul->Add chain must fuse; got elementwise={counts['elementwise']}"

    def test_post_lowering_matmul_relu_fuses(self, runner, tmp_path):
        """After ConvLowering decomposes Conv into sc_low.matmul, the second
        OperatorFusionPass must fuse the downstream Relu into the matmul.
        This verifies the two-pass fusion architecture is working end-to-end."""
        W = numpy_helper.from_array(
            np.zeros((8, 3, 3, 3), dtype=np.float32), "W")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        nodes = [
            helper.make_node("Conv",  ["X","W"],     ["conv_out"],
                              strides=[1,1], pads=[0,0,0,0]),
            helper.make_node("Relu",  ["conv_out"],  ["Y"]),
        ]
        graph = helper.make_graph(nodes, "conv-relu", [X], [Y],
                                   initializer=[W])
        _write(str(tmp_path / "conv_relu.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "conv_relu.onnx"),
            str(tmp_path / "conv_relu.sir"))

        counts = _fusion_counts(summary)
        assert counts["matmul_relu"] >= 1, \
            f"Post-lowering matmul+relu fusion must fire; " \
            f"got matmul_relu={counts['matmul_relu']}"

    @pytest.mark.parametrize("n_pairs", [1, 2, 4])
    def test_n_conv_bn_pairs_each_fuse(self, runner, tmp_path, n_pairs):
        """N sequential Conv+BN pairs must each fuse independently.
        The fusion count must equal n_pairs exactly."""
        inits, nodes = [], []
        C_in = 3

        for i in range(n_pairs):
            C_out = 8
            inits += [
                numpy_helper.from_array(
                    np.zeros((C_out, C_in, 3, 3), dtype=np.float32), f"W{i}"),
                numpy_helper.from_array(np.ones(C_out,  dtype=np.float32), f"sc{i}"),
                numpy_helper.from_array(np.zeros(C_out, dtype=np.float32), f"bi{i}"),
                numpy_helper.from_array(np.zeros(C_out, dtype=np.float32), f"mn{i}"),
                numpy_helper.from_array(np.ones(C_out,  dtype=np.float32), f"vr{i}"),
            ]
            in_name = "X" if i == 0 else f"r{i-1}"
            nodes += [
                helper.make_node("Conv",
                                  [in_name, f"W{i}"], [f"c{i}"],
                                  strides=[1,1], pads=[1,1,1,1]),
                helper.make_node("BatchNormalization",
                                  [f"c{i}", f"sc{i}", f"bi{i}", f"mn{i}", f"vr{i}"],
                                  [f"bn{i}"]),
                helper.make_node("Relu", [f"bn{i}"], [f"r{i}"]),
            ]
            C_in = C_out

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 16, 16])
        Y = helper.make_tensor_value_info(f"r{n_pairs-1}", TensorProto.FLOAT, None)
        graph = helper.make_graph(nodes, f"n-cbr-{n_pairs}", [X], [Y],
                                   initializer=inits)
        _write(str(tmp_path / f"n_cbr_{n_pairs}.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / f"n_cbr_{n_pairs}.onnx"),
            str(tmp_path / f"n_cbr_{n_pairs}.sir"))

        counts = _fusion_counts(summary)
        assert counts["conv_bn"] == n_pairs, \
            f"Expected {n_pairs} conv+bn fusions; got {counts['conv_bn']}"

    def test_fused_ew_op_sequence_in_log(self, runner, tmp_path):
        """After elementwise fusion, the Logger must contain 'fused_ew' or
        'op_sequence', confirming the attribute-encoding path ran."""
        W1 = numpy_helper.from_array(np.ones((2,4), dtype=np.float32), "W1")
        W2 = numpy_helper.from_array(np.ones((2,4), dtype=np.float32), "W2")
        X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
        nodes = [
            helper.make_node("Add", ["X",       "W1"], ["add_out"]),
            helper.make_node("Mul", ["add_out", "W2"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "add-mul", [X], [Y],
                                   initializer=[W1, W2])
        _write(str(tmp_path / "add_mul.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "add_mul.onnx"),
            str(tmp_path / "add_mul.sir"))

        combined = _combined(summary)
        assert "fused_ew" in combined or "op_sequence" in combined, \
            "Logger must mention fused_ew or op_sequence after elementwise fusion"


# ===========================================================================
# 5. DeadCodeEliminationPass — liveness correctness
# ===========================================================================

class TestDCELiveness:

    def test_block_arg_never_removed(self, runner, tmp_path):
        """Block arguments are live-in by definition.
        DCE must never remove them, even when no node consumes them."""
        W = numpy_helper.from_array(np.zeros((4,4), dtype=np.float32), "W")
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 4])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 4])  # unused
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4, 4])
        graph = helper.make_graph(
            [helper.make_node("Add", ["A","W"], ["C"])],
            "unused-input", [A, B], [C], initializer=[W])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(str(tmp_path / "unused_ba.onnx"), "wb") as f:
            f.write(model.SerializeToString())

        summary = runner.run_and_parse(
            str(tmp_path / "unused_ba.onnx"),
            str(tmp_path / "unused_ba.sir"))

        assert summary.returncode == 0, \
            "DCE must not crash when a block argument is unused"

    def test_dead_producer_chain_fully_erased(self, runner, tmp_path):
        """If Add_dead produces dead1, and Mul_dead consumes dead1 to produce
        dead2 (also unused), both ops must be removed. The reverse-order erasure
        in DCE must handle this without a use-def violation."""
        W_dead = numpy_helper.from_array(np.zeros((4,4), dtype=np.float32), "W_dead")
        W_live = numpy_helper.from_array(np.ones((4,4),  dtype=np.float32), "W_live")
        X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4])
        Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])
        nodes = [
            helper.make_node("Add", ["X",     "W_dead"], ["dead1"]),
            helper.make_node("Mul", ["dead1", "W_dead"], ["dead2"]),  # consumes dead1
            helper.make_node("Add", ["X",     "W_live"], ["Y"]),      # live
        ]
        graph = helper.make_graph(nodes, "dead-chain", [X], [Y],
                                   initializer=[W_dead, W_live])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(str(tmp_path / "dead_chain.onnx"), "wb") as f:
            f.write(model.SerializeToString())

        summary = runner.run_and_parse(
            str(tmp_path / "dead_chain.onnx"),
            str(tmp_path / "dead_chain.sir"))

        assert summary.returncode == 0, \
            "DCE must not crash during reverse-order erasure of a dead chain"
        assert _total_dce(summary) >= 2, \
            f"Dead chain (Add_dead + Mul_dead) must remove >= 2 ops; " \
            f"got {_total_dce(summary)}"

    def test_live_op_with_one_dead_user_preserved(self, runner, tmp_path):
        """shared_out feeds one live op and one dead op. DCE must remove only
        the dead branch and keep shared_out's defining op intact."""
        W1 = numpy_helper.from_array(np.zeros((4,4), dtype=np.float32), "W1")
        W2 = numpy_helper.from_array(np.zeros((4,4), dtype=np.float32), "W2")
        X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [4, 4])
        Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [4, 4])
        nodes = [
            helper.make_node("Add", ["X",          "W1"], ["shared_out"]),
            helper.make_node("Add", ["shared_out", "W2"], ["Y"]),            # live
            helper.make_node("Mul", ["shared_out", "W1"], ["dead_branch"]),  # dead
        ]
        graph = helper.make_graph(nodes, "shared-live", [X], [Y],
                                   initializer=[W1, W2])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(str(tmp_path / "shared_live.onnx"), "wb") as f:
            f.write(model.SerializeToString())

        summary = runner.run_and_parse(
            str(tmp_path / "shared_live.onnx"),
            str(tmp_path / "shared_live.sir"))

        assert summary.returncode == 0
        assert _total_dce(summary) == 1, \
            f"Only the dead Mul must be removed; expected total DCE = 1, " \
            f"got {_total_dce(summary)}"

    def test_dce_second_pass_removes_nothing(self, runner, tmp_path):
        """The pipeline runs DCE twice. After the first pass cleans the block,
        the second pass must report 0 removals — DCE must be idempotent."""
        W = numpy_helper.from_array(np.zeros((4,8), dtype=np.float32), "W")
        b = numpy_helper.from_array(np.zeros(8,     dtype=np.float32), "b")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8])
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["X","W","b"], ["Y"])],
            "idem", [X], [Y], initializer=[W, b])
        _write(str(tmp_path / "idem.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "idem.onnx"),
            str(tmp_path / "idem.sir"))

        counts = _dce_counts(summary)
        assert len(counts) >= 2, \
            f"Pipeline must execute DCE >= 2 times; got {len(counts)} entries"
        assert counts[-1] == 0, \
            f"Second DCE pass must be a no-op; got {counts[-1]} removals"


# ===========================================================================
# 6. DeadCodeEliminationPass — edge cases
# ===========================================================================

class TestDCEEdgeCases:

    def test_mostly_dead_block_leaves_one_live_op(self, runner, tmp_path):
        """Three dead Add ops and one live Add op: after DCE the final block
        must contain exactly 1 non-constant op (the live Add)."""
        W_live = numpy_helper.from_array(np.ones((4,4),  dtype=np.float32), "W_live")
        Wd1    = numpy_helper.from_array(np.ones((4,4),  dtype=np.float32), "Wd1")
        Wd2    = numpy_helper.from_array(np.ones((4,4),  dtype=np.float32), "Wd2")
        Wd3    = numpy_helper.from_array(np.ones((4,4),  dtype=np.float32), "Wd3")
        X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4])
        Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 4])
        nodes = [
            helper.make_node("Add", ["X","Wd1"],   ["d1"]),    # dead
            helper.make_node("Add", ["X","Wd2"],   ["d2"]),    # dead
            helper.make_node("Add", ["X","Wd3"],   ["d3"]),    # dead
            helper.make_node("Add", ["X","W_live"],["Y"]),     # live
        ]
        graph = helper.make_graph(nodes, "mostly-dead", [X], [Y],
                                   initializer=[W_live, Wd1, Wd2, Wd3])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(str(tmp_path / "mostly_dead.onnx"), "wb") as f:
            f.write(model.SerializeToString())

        summary = runner.run_and_parse(
            str(tmp_path / "mostly_dead.onnx"),
            str(tmp_path / "mostly_dead.sir"))

        assert summary.returncode == 0
        # 3 dead Add ops removed; dead weight constants may also be removed.
        assert _total_dce(summary) >= 3, \
            f"Expected >= 3 DCE removals; got {_total_dce(summary)}"

    def test_dce_does_not_remove_side_effecting_ops(self, runner, tmp_path):
        """The isMemoryOp() / isControlFlow() guard must prevent sc_mem and
        sc_ctrl ops from being treated as dead. As a proxy test, verify the
        pipeline does not crash on standard sc_high models — the guard must
        not accidentally fire on sc_high ops either."""
        W = numpy_helper.from_array(np.zeros((4,8), dtype=np.float32), "W")
        b = numpy_helper.from_array(np.zeros(8,     dtype=np.float32), "b")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8])
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["X","W","b"], ["Y"])],
            "side-eff", [X], [Y], initializer=[W, b])
        _write(str(tmp_path / "side_eff.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "side_eff.onnx"),
            str(tmp_path / "side_eff.sir"))

        assert summary.returncode == 0, \
            "DCE side-effect guard must not misfire on sc_high ops"


# ===========================================================================
# 7. Pass interaction — fusion + DCE composition
# ===========================================================================

class TestPassInteraction:

    def test_conv_bn_fusion_dce_removes_exactly_one(self, runner, tmp_path):
        """Conv+BN (no Relu): fusion absorbs BN into Conv attributes.
        The first DCE pass must then remove exactly the dead BN op = 1."""
        W     = numpy_helper.from_array(
            np.zeros((8,3,3,3), dtype=np.float32), "W")
        scale = numpy_helper.from_array(np.ones(8,  dtype=np.float32), "scale")
        bias  = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "bias")
        mean  = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "mean")
        var   = numpy_helper.from_array(np.ones(8,  dtype=np.float32), "var")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        nodes = [
            helper.make_node("Conv", ["X","W"], ["conv_out"],
                              strides=[1,1], pads=[1,1,1,1]),
            helper.make_node("BatchNormalization",
                              ["conv_out","scale","bias","mean","var"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "conv-bn-only", [X], [Y],
                                   initializer=[W, scale, bias, mean, var])
        _write(str(tmp_path / "conv_bn_only.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "conv_bn_only.onnx"),
            str(tmp_path / "conv_bn_only.sir"))

        assert _fusion_counts(summary)["conv_bn"] == 1
        dce_passes = _dce_counts(summary)
        assert dce_passes[0] == 1, \
            f"First DCE pass must remove exactly the dead BN op; " \
            f"got {dce_passes[0]}"

    def test_double_conv_bn_relu_total_counts(self, runner, tmp_path):
        """Two sequential Conv+BN+Relu blocks must yield:
          conv_bn fusions = 2, lowering count = 2, total DCE >= 2."""
        def _cbr_inits(i, C_in, C_out):
            return [
                numpy_helper.from_array(
                    np.zeros((C_out,C_in,3,3), dtype=np.float32), f"W{i}"),
                numpy_helper.from_array(np.ones(C_out,  dtype=np.float32), f"s{i}"),
                numpy_helper.from_array(np.zeros(C_out, dtype=np.float32), f"b{i}"),
                numpy_helper.from_array(np.zeros(C_out, dtype=np.float32), f"m{i}"),
                numpy_helper.from_array(np.ones(C_out,  dtype=np.float32), f"v{i}"),
            ]

        inits  = _cbr_inits(0, 3, 8) + _cbr_inits(1, 8, 16)
        nodes  = [
            helper.make_node("Conv", ["X","W0"],["c0"],strides=[1,1],pads=[1,1,1,1]),
            helper.make_node("BatchNormalization",["c0","s0","b0","m0","v0"],["bn0"]),
            helper.make_node("Relu", ["bn0"], ["r0"]),
            helper.make_node("Conv", ["r0","W1"],["c1"],strides=[1,1],pads=[1,1,1,1]),
            helper.make_node("BatchNormalization",["c1","s1","b1","m1","v1"],["bn1"]),
            helper.make_node("Relu", ["bn1"], ["Y"]),
        ]
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1,3,8,8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        graph = helper.make_graph(nodes, "double-cbr", [X], [Y], initializer=inits)
        _write(str(tmp_path / "double_cbr.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "double_cbr.onnx"),
            str(tmp_path / "double_cbr.sir"))

        counts = _fusion_counts(summary)
        assert counts["conv_bn"]   == 2, \
            f"Expected 2 conv+bn; got {counts['conv_bn']}"
        assert _lowering_count(summary) == 2, \
            "Both Conv ops must be lowered"
        assert _total_dce(summary) >= 2, \
            f"Total DCE must be >= 2; got {_total_dce(summary)}"

    def test_conv_relu_full_pass_sequence_counts(self, runner, tmp_path):
        """Conv -> Relu exercises the complete 5-pass sequence:
          [1] fusion: 0 conv+bn (no BN), 0 matmul+relu (not lowered yet)
          [2] DCE:    0 removals
          [3] lower:  conv -> im2col + reshape + matmul  (lowering = 1)
          [4] fusion: matmul + relu -> matmul(activation=relu) (matmul_relu = 1)
          [5] DCE:    relu op removed (DCE last pass >= 1)
        Assert each count individually to pinpoint any regression."""
        W = numpy_helper.from_array(
            np.zeros((8,3,3,3), dtype=np.float32), "W")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1,3,8,8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        nodes = [
            helper.make_node("Conv",  ["X","W"], ["conv_out"],
                              strides=[1,1], pads=[0,0,0,0]),
            helper.make_node("Relu",  ["conv_out"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "conv-relu-seq", [X], [Y],
                                   initializer=[W])
        _write(str(tmp_path / "conv_relu_seq.onnx"),
               helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "conv_relu_seq.onnx"),
            str(tmp_path / "conv_relu_seq.sir"))

        assert _lowering_count(summary) == 1, \
            "Conv must be lowered exactly once"
        counts = _fusion_counts(summary)
        assert counts["matmul_relu"] >= 1, \
            f"Post-lowering Relu must fuse into matmul; got {counts['matmul_relu']}"
        assert _total_dce(summary) >= 1, \
            f"Relu op must be cleaned up by DCE; total DCE = {_total_dce(summary)}"
        assert summary.returncode == 0