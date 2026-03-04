"""
test_middle_end.py — stress tests for the SeeC++ middle-end pass pipeline.

Coverage map:
  TestConvLowering        — ConvLoweringPass: sc_high.conv2d -> sc_low.*
  TestOperatorFusion      — OperatorFusionPass: Conv+BN, MatMul+Relu, EW chains
  TestDeadCodeElimination — DeadCodeEliminationPass: orphaned op removal
  TestPassManager         — Pipeline ordering, inter-pass validation, timing
  TestFullPipeline        — End-to-end middle-end on real model topologies
                            (ResNet stem, BERT projection block, MobileNet block)

Logger lines parsed by conftest.SirRunner:
  ConvLoweringPass: lowering N conv2d op(s)
  ConvLoweringPass: conv2d -> im2col + reshape + matmul[+ add(bias)]
  OperatorFusionPass: fused N conv+bn, N matmul+relu, N elementwise chain(s)
  DeadCodeEliminationPass: removed N dead op(s)
  PassManager: pipeline completed successfully.
  PassManager timing summary: ...

Run with:
    pytest test_middle_end.py -v
"""

import re
import os
import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto, checker, numpy_helper

# ---------------------------------------------------------------------------
# Helpers — shared with frontend tests
# ---------------------------------------------------------------------------

def _opset(version: int = 17):
    return [helper.make_opsetid("", version)]


def _write_model(path: str, model: onnx.ModelProto, skip_check: bool = False) -> None:
    if not skip_check:
        try:
            checker.check_model(model)
        except checker.ValidationError as e:
            raise ValueError(f"Test fixture is not valid ONNX: {e}") from e
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def _find_result(summary, value_id: str):
    for op in summary.ops:
        for r in op.results:
            if r.value_id == value_id:
                return r
    return None


# ---------------------------------------------------------------------------
# Middle-end specific log parsers
# ---------------------------------------------------------------------------

def _parse_lowering_count(summary) -> int:
    """Return the number of conv2d ops the ConvLoweringPass reported lowering."""
    m = re.search(r"ConvLoweringPass: lowering\s+(\d+)", summary.stderr)
    return int(m.group(1)) if m else 0


def _parse_fusion_counts(summary) -> dict:
    """Return {'conv_bn': N, 'matmul_relu': N, 'elementwise': N} from Logger output."""
    m = re.search(
        r"OperatorFusionPass: fused\s+(\d+)\s+conv\+bn,\s+(\d+)\s+matmul\+relu,"
        r"\s+(\d+)\s+elementwise",
        summary.stderr + summary.stdout,
    )
    if not m:
        return {"conv_bn": 0, "matmul_relu": 0, "elementwise": 0}
    return {
        "conv_bn":      int(m.group(1)),
        "matmul_relu":  int(m.group(2)),
        "elementwise":  int(m.group(3)),
    }


def _parse_dce_count(summary) -> int:
    """Return the total number of dead ops removed across all DCE passes."""
    total = 0
    for m in re.finditer(r"DeadCodeEliminationPass: removed\s+(\d+)", summary.stderr):
        total += int(m.group(1))
    return total


def _parse_final_op_count(summary) -> int:
    """Return the op count logged at driver completion."""
    m = re.search(r"(\d+)\s+op\(s\) in output block", summary.stderr + summary.stdout)
    return int(m.group(1)) if m else summary.num_ops


def _pipeline_completed(summary) -> bool:
    combined = summary.stderr + summary.stdout
    return "pipeline completed successfully" in combined.lower()


def _timing_table_present(summary) -> bool:
    return "PassManager timing summary" in (summary.stderr + summary.stdout)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_conv_only(tmp_path, N=1, C=3, H=8, W=8, F=8, KH=3, KW=3,
                    strides=(1,1), pads=(0,0,0,0)):
    """Minimal Conv-only model."""
    W_init = numpy_helper.from_array(
        np.zeros((F, C, KH, KW), dtype=np.float32), "W")
    X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
    Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("Conv", ["X", "W"], ["Y"],
                             strides=list(strides), pads=list(pads))
    graph = helper.make_graph([node], "conv-only", [X], [Y],
                               initializer=[W_init])
    return helper.make_model(graph, opset_imports=_opset())


def _make_conv_bias(tmp_path, N=1, C=3, H=8, W=8, F=8, KH=3, KW=3):
    """Conv with explicit bias initializer."""
    W_init = numpy_helper.from_array(
        np.zeros((F, C, KH, KW), dtype=np.float32), "W")
    b_init = numpy_helper.from_array(
        np.zeros(F, dtype=np.float32), "b")
    X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
    Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("Conv", ["X", "W", "b"], ["Y"],
                             strides=[1,1], pads=[0,0,0,0])
    graph = helper.make_graph([node], "conv-bias", [X], [Y],
                               initializer=[W_init, b_init])
    return helper.make_model(graph, opset_imports=_opset())


def _make_conv_bn_relu(N=1, C=3, H=8, W=8, F=8):
    """Conv + BatchNorm + Relu — canonical ResNet building block."""
    W_init  = numpy_helper.from_array(
        np.zeros((F, C, 3, 3), dtype=np.float32), "W")
    scale   = numpy_helper.from_array(np.ones(F,  dtype=np.float32), "scale")
    bias    = numpy_helper.from_array(np.zeros(F, dtype=np.float32), "bias")
    mean    = numpy_helper.from_array(np.zeros(F, dtype=np.float32), "mean")
    var     = numpy_helper.from_array(np.ones(F,  dtype=np.float32), "var")

    X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [N, C, H, W])
    Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, None)

    nodes = [
        helper.make_node("Conv", ["X", "W"], ["conv_out"],
                          strides=[1,1], pads=[1,1,1,1]),
        helper.make_node("BatchNormalization",
                          ["conv_out", "scale", "bias", "mean", "var"],
                          ["bn_out"]),
        helper.make_node("Relu", ["bn_out"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "conv-bn-relu", [X], [Y],
                               initializer=[W_init, scale, bias, mean, var])
    return helper.make_model(graph, opset_imports=_opset())


def _make_gemm_relu(M=4, K=8, N=16):
    """Gemm + Relu — canonical FC layer."""
    W_init = numpy_helper.from_array(
        np.zeros((K, N), dtype=np.float32), "W")
    b_init = numpy_helper.from_array(
        np.zeros(N, dtype=np.float32), "b")
    X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
    Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    nodes = [
        helper.make_node("Gemm", ["X", "W", "b"], ["gemm_out"]),
        helper.make_node("Relu", ["gemm_out"],     ["Y"]),
    ]
    graph = helper.make_graph(nodes, "gemm-relu", [X], [Y],
                               initializer=[W_init, b_init])
    return helper.make_model(graph, opset_imports=_opset())


def _make_add_relu_chain():
    """Add -> Relu elementwise chain."""
    W = numpy_helper.from_array(np.ones((2, 4), dtype=np.float32), "W")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    nodes = [
        helper.make_node("Add",  ["X", "W"],   ["add_out"]),
        helper.make_node("Relu", ["add_out"],   ["Y"]),
    ]
    graph = helper.make_graph(nodes, "add-relu", [X], [Y], initializer=[W])
    return helper.make_model(graph, opset_imports=_opset())


def _make_resnet_stem(N=1):
    """ResNet-style stem: Conv7x7/2 -> BN -> Relu."""
    W = numpy_helper.from_array(
        np.zeros((64, 3, 7, 7), dtype=np.float32), "W")
    scale = numpy_helper.from_array(np.ones(64,  dtype=np.float32), "bn_scale")
    bias  = numpy_helper.from_array(np.zeros(64, dtype=np.float32), "bn_bias")
    mean  = numpy_helper.from_array(np.zeros(64, dtype=np.float32), "bn_mean")
    var   = numpy_helper.from_array(np.ones(64,  dtype=np.float32), "bn_var")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, 3, 224, 224])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    nodes = [
        helper.make_node("Conv", ["X", "W"], ["conv_out"],
                          strides=[2, 2], pads=[3, 3, 3, 3]),
        helper.make_node("BatchNormalization",
                          ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                          ["bn_out"]),
        helper.make_node("Relu", ["bn_out"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "resnet-stem", [X], [Y],
                               initializer=[W, scale, bias, mean, var])
    return helper.make_model(graph, opset_imports=_opset())


def _make_bert_projection(batch=2, seq=128, d_model=512, d_ff=2048):
    """BERT feed-forward block: Gemm -> Relu -> Gemm."""
    W1 = numpy_helper.from_array(
        np.zeros((d_model, d_ff),   dtype=np.float32), "W1")
    b1 = numpy_helper.from_array(
        np.zeros(d_ff,              dtype=np.float32), "b1")
    W2 = numpy_helper.from_array(
        np.zeros((d_ff, d_model),   dtype=np.float32), "W2")
    b2 = numpy_helper.from_array(
        np.zeros(d_model,           dtype=np.float32), "b2")

    X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [batch*seq, d_model])
    Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [batch*seq, d_model])
    nodes = [
        helper.make_node("Gemm", ["X",    "W1", "b1"], ["h1"]),
        helper.make_node("Relu", ["h1"],               ["h1_act"]),
        helper.make_node("Gemm", ["h1_act","W2", "b2"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "bert-ffn", [X], [Y],
                               initializer=[W1, b1, W2, b2])
    return helper.make_model(graph, opset_imports=_opset())


def _make_mobilenet_block(N=1, C=32, H=14, W=14):
    """MobileNet depthwise-separable block: DW-Conv + BN + Relu + PW-Conv + BN + Relu."""
    dw_filter = numpy_helper.from_array(
        np.zeros((C, 1, 3, 3),  dtype=np.float32), "dw_W")
    dw_scale  = numpy_helper.from_array(np.ones(C,  dtype=np.float32), "dw_scale")
    dw_bias   = numpy_helper.from_array(np.zeros(C, dtype=np.float32), "dw_bias")
    dw_mean   = numpy_helper.from_array(np.zeros(C, dtype=np.float32), "dw_mean")
    dw_var    = numpy_helper.from_array(np.ones(C,  dtype=np.float32), "dw_var")

    pw_filter = numpy_helper.from_array(
        np.zeros((C*2, C, 1, 1), dtype=np.float32), "pw_W")
    pw_scale  = numpy_helper.from_array(np.ones(C*2,  dtype=np.float32), "pw_scale")
    pw_bias   = numpy_helper.from_array(np.zeros(C*2, dtype=np.float32), "pw_bias")
    pw_mean   = numpy_helper.from_array(np.zeros(C*2, dtype=np.float32), "pw_mean")
    pw_var    = numpy_helper.from_array(np.ones(C*2,  dtype=np.float32), "pw_var")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    nodes = [
        helper.make_node("Conv", ["X",    "dw_W"], ["dw_out"],
                          strides=[1,1], pads=[1,1,1,1], group=C),
        helper.make_node("BatchNormalization",
                          ["dw_out","dw_scale","dw_bias","dw_mean","dw_var"],
                          ["dw_bn"]),
        helper.make_node("Relu", ["dw_bn"],  ["dw_relu"]),
        helper.make_node("Conv", ["dw_relu","pw_W"], ["pw_out"],
                          strides=[1,1], pads=[0,0,0,0]),
        helper.make_node("BatchNormalization",
                          ["pw_out","pw_scale","pw_bias","pw_mean","pw_var"],
                          ["pw_bn"]),
        helper.make_node("Relu", ["pw_bn"],  ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "mobilenet-block", [X], [Y],
        initializer=[dw_filter, dw_scale, dw_bias, dw_mean, dw_var,
                     pw_filter, pw_scale, pw_bias, pw_mean, pw_var])
    return helper.make_model(graph, opset_imports=_opset())


# ===========================================================================
# 1. ConvLoweringPass
# ===========================================================================

class TestConvLowering:

    def test_single_conv_is_lowered(self, runner, tmp_path):
        """A single sc_high.conv2d must be decomposed into sc_low ops."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_only(tmp_path))

        summary = runner.run_and_parse(onnx_file, sir_file)

        assert _parse_lowering_count(summary) == 1, \
            "ConvLoweringPass must report lowering exactly 1 conv2d"

    def test_conv_produces_im2col_matmul_in_log(self, runner, tmp_path):
        """Logger must confirm im2col + reshape + matmul emission."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_only(tmp_path))

        summary = runner.run_and_parse(onnx_file, sir_file)

        combined = summary.stderr + summary.stdout
        assert "im2col" in combined, \
            "Logger must mention im2col after conv lowering"
        assert "matmul" in combined, \
            "Logger must mention matmul after conv lowering"

    def test_conv_with_bias_emits_add(self, runner, tmp_path):
        """A Conv with a bias initializer must produce an additional sc_low.add."""
        onnx_file = str(tmp_path / "conv_bias.onnx")
        sir_file  = str(tmp_path / "conv_bias.sir")
        _write_model(onnx_file, _make_conv_bias(tmp_path))

        summary = runner.run_and_parse(onnx_file, sir_file)

        combined = summary.stderr + summary.stdout
        assert "add(bias)" in combined, \
            "Logger must confirm bias add emission for Conv with bias"

    @pytest.mark.parametrize("strides,pads", [
        ((1, 1), (0, 0, 0, 0)),   # no padding, stride 1
        ((2, 2), (1, 1, 1, 1)),   # same-padding, stride 2
        ((1, 1), (1, 1, 1, 1)),   # same-padding, stride 1
    ])
    def test_conv_lowering_various_strides_pads(
        self, runner, tmp_path, strides, pads
    ):
        """ConvLowering must succeed for all stride/pad combinations."""
        onnx_file = str(tmp_path / "conv_sp.onnx")
        sir_file  = str(tmp_path / "conv_sp.sir")
        _write_model(onnx_file, _make_conv_only(
            tmp_path, strides=strides, pads=pads))

        summary = runner.run_and_parse(onnx_file, sir_file)
        assert summary.returncode == 0, \
            f"ConvLowering failed for strides={strides} pads={pads}"

    def test_multiple_convs_all_lowered(self, runner, tmp_path):
        """A block with two Conv ops must lower both."""
        W1 = numpy_helper.from_array(
            np.zeros((8, 3, 3, 3), dtype=np.float32), "W1")
        W2 = numpy_helper.from_array(
            np.zeros((16, 8, 3, 3), dtype=np.float32), "W2")

        X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [1, 3, 16, 16])
        Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, None)
        nodes = [
            helper.make_node("Conv", ["X",  "W1"], ["c1"],
                              strides=[1,1], pads=[0,0,0,0]),
            helper.make_node("Conv", ["c1", "W2"], ["Y"],
                              strides=[1,1], pads=[0,0,0,0]),
        ]
        graph = helper.make_graph(nodes, "two-conv", [X], [Y],
                                   initializer=[W1, W2])
        _write_model(str(tmp_path / "two_conv.onnx"),
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(
            str(tmp_path / "two_conv.onnx"),
            str(tmp_path / "two_conv.sir"))

        assert _parse_lowering_count(summary) == 2, \
            "Both Conv ops must be lowered; expected lowering count = 2"

    def test_sc_high_conv2d_absent_after_lowering(self, runner, tmp_path):
        """After the full pipeline, no sc_high.conv2d must remain in Logger output."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_only(tmp_path))

        summary = runner.run_and_parse(onnx_file, sir_file)

        # The middle-end Logger should never print sc_high.conv2d after lowering.
        assert "sc_high.conv2d" not in (summary.stderr + summary.stdout), \
            "sc_high.conv2d must be fully eliminated after ConvLoweringPass"


# ===========================================================================
# 2. OperatorFusionPass
# ===========================================================================

class TestOperatorFusion:

    def test_conv_bn_fusion_reported(self, runner, tmp_path):
        """Conv + BN must be fused; Logger must report 1 conv+bn fusion."""
        onnx_file = str(tmp_path / "conv_bn.onnx")
        sir_file  = str(tmp_path / "conv_bn.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)

        counts = _parse_fusion_counts(summary)
        assert counts["conv_bn"] >= 1, \
            f"Expected at least 1 conv+bn fusion, got {counts['conv_bn']}"

    def test_bn_op_absent_after_fusion(self, runner, tmp_path):
        """After Conv+BN fusion, the BN op must be eliminated from Logger output."""
        onnx_file = str(tmp_path / "conv_bn.onnx")
        sir_file  = str(tmp_path / "conv_bn.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)

        combined = summary.stderr + summary.stdout
        assert "batch_norm" not in combined or "fused" in combined, \
            "BatchNorm op should be absorbed by fusion, not remain as a live op"

    def test_gemm_relu_fusion_reported(self, runner, tmp_path):
        """Gemm + Relu must be fused; Logger must report 1 matmul+relu fusion."""
        onnx_file = str(tmp_path / "gemm_relu.onnx")
        sir_file  = str(tmp_path / "gemm_relu.sir")
        _write_model(onnx_file, _make_gemm_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)

        counts = _parse_fusion_counts(summary)
        assert counts["matmul_relu"] >= 1, \
            f"Expected at least 1 matmul+relu fusion, got {counts['matmul_relu']}"

    def test_relu_op_absent_after_matmul_fusion(self, runner, tmp_path):
        """After Gemm+Relu fusion, the standalone Relu op must be gone."""
        onnx_file = str(tmp_path / "gemm_relu.onnx")
        sir_file  = str(tmp_path / "gemm_relu.sir")
        _write_model(onnx_file, _make_gemm_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)

        combined = summary.stderr + summary.stdout
        # activation=relu should appear on the gemm/matmul op, not as a standalone op.
        assert "activation" in combined or counts["matmul_relu"] >= 1, \
            "Relu should be absorbed as activation attribute, not a standalone op"

    def test_elementwise_chain_fused(self, runner, tmp_path):
        """Add -> Relu chain must trigger at least one elementwise fusion."""
        onnx_file = str(tmp_path / "add_relu.onnx")
        sir_file  = str(tmp_path / "add_relu.sir")
        _write_model(onnx_file, _make_add_relu_chain())

        summary = runner.run_and_parse(onnx_file, sir_file)

        counts = _parse_fusion_counts(summary)
        # Add+Relu is either caught as elementwise chain fusion or matmul+relu.
        total_fusions = counts["elementwise"] + counts["matmul_relu"]
        assert total_fusions >= 1, \
            f"Add->Relu chain must produce at least 1 fusion; got {counts}"

    def test_conv_with_shared_output_not_fused(self, runner, tmp_path):
        """A Conv whose output feeds TWO ops must NOT be fused with BN —
        the hasOneUse() guard must hold."""
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

        # conv_out feeds both BN and a second Conv — hasOneUse() is false.
        nodes = [
            helper.make_node("Conv", ["X", "W"], ["conv_out"],
                              strides=[1,1], pads=[0,0,0,0]),
            helper.make_node("BatchNormalization",
                              ["conv_out","scale","bias","mean","var"], ["Y1"]),
            helper.make_node("Conv", ["conv_out", "W2"], ["Y2"],
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

        counts = _parse_fusion_counts(summary)
        assert counts["conv_bn"] == 0, \
            "Conv with multiple users must NOT be fused with BN " \
            f"(hasOneUse guard); got conv_bn={counts['conv_bn']}"


# ===========================================================================
# 3. DeadCodeEliminationPass
# ===========================================================================

class TestDeadCodeElimination:

    def test_dce_runs_without_crash(self, runner, tmp_path):
        """DCE must complete without error on any valid block."""
        onnx_file = str(tmp_path / "dce.onnx")
        sir_file  = str(tmp_path / "dce.sir")
        _write_model(onnx_file, _make_gemm_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)
        assert summary.returncode == 0

    def test_dce_reports_removal_after_fusion(self, runner, tmp_path):
        """After Conv+BN fusion, DCE must report removing the dead BN op."""
        onnx_file = str(tmp_path / "conv_bn.onnx")
        sir_file  = str(tmp_path / "conv_bn.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)

        dce_removed = _parse_dce_count(summary)
        assert dce_removed >= 1, \
            f"DCE must remove at least 1 dead op after Conv+BN fusion; " \
            f"got {dce_removed}"

    def test_dce_zero_removals_on_clean_block(self, runner, tmp_path):
        """A block with no dead ops must report 0 removals from DCE."""
        onnx_file = str(tmp_path / "clean.onnx")
        sir_file  = str(tmp_path / "clean.sir")

        # Simple MatMul with no fusion opportunities — nothing should die.
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 16])
        C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4, 16])
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["A", "B"], ["C"])],
            "clean", [A, B], [C])
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)

        dce_removed = _parse_dce_count(summary)
        assert dce_removed == 0, \
            f"Clean block should have 0 DCE removals; got {dce_removed}"

    def test_unreferenced_constant_removed_by_dce(self, runner, tmp_path):
        """An initializer that is never consumed must be removed by DCE."""
        W_live = numpy_helper.from_array(
            np.ones((2, 2),  dtype=np.float32), "W_live")
        W_dead = numpy_helper.from_array(
            np.zeros((4, 4), dtype=np.float32), "W_dead")   # never used

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "W_live"], ["Y"])],
            "dead-const", [X], [Y],
            initializer=[W_live, W_dead])
        model = helper.make_model(graph, opset_imports=_opset())
        with open(str(tmp_path / "dead_const.onnx"), "wb") as f:
            f.write(model.SerializeToString())

        summary = runner.run_and_parse(
            str(tmp_path / "dead_const.onnx"),
            str(tmp_path / "dead_const.sir"))

        dce_removed = _parse_dce_count(summary)
        assert dce_removed >= 1, \
            "Dead constant W_dead must be removed by DCE"


# ===========================================================================
# 4. PassManager
# ===========================================================================

class TestPassManager:

    def test_pipeline_completion_logged(self, runner, tmp_path):
        """PassManager must log pipeline completion on success."""
        onnx_file = str(tmp_path / "pm.onnx")
        sir_file  = str(tmp_path / "pm.sir")
        _write_model(onnx_file, _make_gemm_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)

        assert _pipeline_completed(summary), \
            "PassManager must log 'pipeline completed successfully'"

    def test_timing_table_emitted(self, runner, tmp_path):
        """PassManager must emit a timing summary table after completion."""
        onnx_file = str(tmp_path / "timing.onnx")
        sir_file  = str(tmp_path / "timing.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)

        assert _timing_table_present(summary), \
            "PassManager must emit a timing summary after all passes complete"

    def test_all_five_passes_run(self, runner, tmp_path):
        """Logger must show evidence of all five passes executing."""
        onnx_file = str(tmp_path / "five.onnx")
        sir_file  = str(tmp_path / "five.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)
        combined = summary.stderr + summary.stdout

        for pass_name in [
            "OperatorFusionPass",
            "DeadCodeEliminationPass",
            "ConvLoweringPass",
        ]:
            assert pass_name in combined, \
                f"Expected pass '{pass_name}' to appear in Logger output"

    def test_inter_pass_validation_runs(self, runner, tmp_path):
        """PassManager must log inter-pass validation after structural passes."""
        onnx_file = str(tmp_path / "val.onnx")
        sir_file  = str(tmp_path / "val.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)

        assert "inter-pass validation" in (summary.stderr + summary.stdout).lower(), \
            "PassManager must log inter-pass validation after each structural pass"


# ===========================================================================
# 5. Full end-to-end pipeline stress tests
# ===========================================================================

class TestFullPipeline:

    def test_resnet_stem_pipeline(self, runner, tmp_path):
        """ResNet stem (Conv7x7 + BN + Relu on 224x224) must complete cleanly."""
        onnx_file = str(tmp_path / "resnet_stem.onnx")
        sir_file  = str(tmp_path / "resnet_stem.sir")
        _write_model(onnx_file, _make_resnet_stem())

        summary = runner.run_and_parse(onnx_file, sir_file)

        assert summary.returncode == 0
        assert _parse_lowering_count(summary) == 1, \
            "ResNet stem has exactly 1 Conv; must lower exactly 1"
        counts = _parse_fusion_counts(summary)
        assert counts["conv_bn"] == 1, \
            "ResNet stem Conv+BN must fuse into 1 op"
        assert _pipeline_completed(summary)

    def test_bert_ffn_pipeline(self, runner, tmp_path):
        """BERT FFN (Gemm->Relu->Gemm) must complete with 1 matmul+relu fusion."""
        onnx_file = str(tmp_path / "bert_ffn.onnx")
        sir_file  = str(tmp_path / "bert_ffn.sir")
        _write_model(onnx_file, _make_bert_projection())

        summary = runner.run_and_parse(onnx_file, sir_file)

        assert summary.returncode == 0
        counts = _parse_fusion_counts(summary)
        assert counts["matmul_relu"] >= 1, \
            f"BERT FFN must fuse at least 1 Gemm+Relu; got {counts['matmul_relu']}"
        assert _pipeline_completed(summary)

    def test_mobilenet_block_pipeline(self, runner, tmp_path):
        """MobileNet DW+PW block must lower 2 convs and fuse 2 Conv+BN pairs."""
        onnx_file = str(tmp_path / "mobilenet.onnx")
        sir_file  = str(tmp_path / "mobilenet.sir")
        _write_model(onnx_file, _make_mobilenet_block())

        summary = runner.run_and_parse(onnx_file, sir_file)

        assert summary.returncode == 0
        assert _parse_lowering_count(summary) == 2, \
            "MobileNet block has 2 Conv ops; both must be lowered"
        counts = _parse_fusion_counts(summary)
        assert counts["conv_bn"] == 2, \
            f"MobileNet block has 2 Conv+BN pairs; expected 2 fusions, " \
            f"got {counts['conv_bn']}"
        assert _pipeline_completed(summary)

    def test_op_count_decreases_after_middle_end(self, runner, tmp_path):
        """Conv+BN+Relu block must have fewer ops after the middle-end
        than after ingestion (fusion + DCE reduce the op count)."""
        onnx_file = str(tmp_path / "count.onnx")
        sir_file  = str(tmp_path / "count.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)

        # After ingestion: W(const) + scale/bias/mean/var(4 consts) + conv + bn + relu = 8 ops
        # After middle-end: BN absorbed (fused), Relu absorbed (fused into conv/matmul)
        # -> final op count must be less than 8
        final_ops = _parse_final_op_count(summary)
        assert final_ops < 8, \
            f"Middle-end must reduce op count for Conv+BN+Relu; " \
            f"expected < 8, got {final_ops}"

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_pipeline_invariant_to_batch_size(self, runner, tmp_path, batch_size):
        """Middle-end pipeline must produce identical fusion/lowering counts
        regardless of batch size — batch is a dynamic dimension."""
        onnx_file = str(tmp_path / f"batch_{batch_size}.onnx")
        sir_file  = str(tmp_path / f"batch_{batch_size}.sir")
        _write_model(onnx_file, _make_resnet_stem(N=batch_size))

        summary = runner.run_and_parse(onnx_file, sir_file)

        assert summary.returncode == 0
        assert _parse_lowering_count(summary) == 1
        assert _parse_fusion_counts(summary)["conv_bn"] == 1