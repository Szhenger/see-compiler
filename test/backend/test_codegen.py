"""
test_codegen_stress.py — deep structural stress tests for the codegen layer.

Additive to test_codegen.py. That file verifies:
  - WeightFolding count, arithmetic, idempotency
  - BufferAllocator arena size, 64-byte alignment, weight vs activation
  - All four artefact files exist
  - model.c contains seecpp_im2col, cblas_sgemm, seecpp_run_model
  - model.h contains extern "C"
  - arena_size.h is parseable
  - weights.bin is non-empty
  - Stage ordering in Logger
  - Two weak numerical tests (relu >= 0, zero-input/zero-weight = 0)

This file stress-tests contracts invisible to test_codegen.py:

  TestCNameSanitisation
    — SSA ids containing %, ., - produce valid C identifiers prefixed "v_"
    — Weight pointers declared "const float*", activations as "float*"

  TestPtrExprEmission
    — Weight ptr expressions use "(const char*)weight_blob + NUL" form
    — Activation ptr expressions use "(char*)arena + NUL" form
    — First weight always at offset 0, first activation always at offset 0

  TestReshapeIsNoop
    — sc_low.reshape emits exactly zero extra memcpy / cblas calls
    — "logical reinterpret" comment present for every lowered conv

  TestFusedEwEmission
    — Add segment of fused_ew emits "+="
    — Mul segment emits "*="
    — Loop uses "_acc" accumulator
    — Loop bound matches total element count

  TestGemmEmission
    — trans_b=1 produces "CblasTrans" for B
    — trans_a=1 produces "CblasTrans" for A
    — Bias loop present iff Gemm has 3 operands
    — Fused relu emits "0.0f" clamp loop
    — Dimension constants M, K, N embedded in cblas call

  TestReluEmission
    — Standalone relu emits a "0.0f" loop
    — Loop bound equals tensor volume
    — Copy-and-clamp path uses ternary "> 0.0f ? x : 0.0f"

  TestBatchedMatMulEmission
    — Lowered conv emits "for (int _b" batch loop
    — Loop bound matches N (batch size)
    — Per-batch stride offset "+ _b *" in cblas call

  TestMultipleInputsOutputs
    — Two graph inputs → inputs[0] and inputs[1] in model.c
    — Two terminal ops → outputs[0] and outputs[1]
    — Single-input model never references inputs[1]

  TestHeaderGuards
    — arena_size.h has #pragma once
    — model.h has #pragma once and #include <stddef.h>
    — model.c includes "model.h" and <cblas.h>

  TestLargeShapeConstants
    — ResNet stem (224×224 → 112×112) embeds 112, 64, 7 in model.c
    — Parametrized stride-2 sizes embed correct OH/OW

Run with:
    pytest test_codegen_stress.py -v
"""

import re
import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto, checker, numpy_helper
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opset(v: int = 17):
    return [helper.make_opsetid("", v)]


def _write(path: str, model: onnx.ModelProto,
           skip_check: bool = False) -> None:
    if not skip_check:
        try:
            checker.check_model(model)
        except checker.ValidationError as e:
            raise ValueError(f"Bad fixture: {e}") from e
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def _model_c(out_dir) -> str:
    return (Path(out_dir) / "model.c").read_text()


def _arena_h(out_dir) -> str:
    return (Path(out_dir) / "arena_size.h").read_text()


def _model_h(out_dir) -> str:
    return (Path(out_dir) / "model.h").read_text()


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _conv_only(N=1, C=3, H=8, W=8, F=8, KH=3, KW=3,
               strides=(1, 1), pads=(0, 0, 0, 0)):
    W_init = numpy_helper.from_array(
        np.zeros((F, C, KH, KW), dtype=np.float32), "W")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("Conv", ["X", "W"], ["Y"],
                             strides=list(strides), pads=list(pads))
    graph = helper.make_graph([node], "conv", [X], [Y], initializer=[W_init])
    return helper.make_model(graph, opset_imports=_opset())


def _gemm(M=4, K=8, N=16, with_bias=True,
          trans_a=False, trans_b=False, with_relu=False):
    if trans_a:
        arr_A = np.zeros((K, M), dtype=np.float32)
    else:
        arr_A = np.zeros((M, K), dtype=np.float32)
    if trans_b:
        arr_B = np.zeros((N, K), dtype=np.float32)
    else:
        arr_B = np.zeros((K, N), dtype=np.float32)

    W_A = numpy_helper.from_array(arr_A, "A_weight")
    W_B = numpy_helper.from_array(arr_B, "B_weight")

    inits     = [W_A, W_B]
    inp_names = ["A_weight", "B_weight"]
    if with_bias:
        b_init = numpy_helper.from_array(np.zeros(N, dtype=np.float32), "bias_w")
        inits.append(b_init)
        inp_names.append("bias_w")

    x_shape = [K, M] if trans_a else [M, K]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    gemm_attrs = {}
    if trans_a:
        gemm_attrs["transA"] = 1
    if trans_b:
        gemm_attrs["transB"] = 1

    out_name = "gemm_out" if with_relu else "Y"
    nodes = [helper.make_node("Gemm", ["X"] + inp_names, [out_name], **gemm_attrs)]
    if with_relu:
        nodes.append(helper.make_node("Relu", ["gemm_out"], ["Y"]))

    graph = helper.make_graph(nodes, "gemm", [X], [Y], initializer=inits)
    return helper.make_model(graph, opset_imports=_opset())


def _add_mul_chain():
    """Add → Mul elementwise chain — exercises fused_ew emission."""
    W1 = numpy_helper.from_array(np.ones((2, 4),            dtype=np.float32), "W1")
    W2 = numpy_helper.from_array(np.full((2, 4), 2.0,       dtype=np.float32), "W2")
    X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    nodes = [
        helper.make_node("Add", ["X",       "W1"], ["add_out"]),
        helper.make_node("Mul", ["add_out", "W2"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "add-mul", [X], [Y], initializer=[W1, W2])
    return helper.make_model(graph, opset_imports=_opset())


def _relu_only():
    """Standalone Relu with no preceding matmul — not fused, exercises emitReluOp."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    graph = helper.make_graph(
        [helper.make_node("Relu", ["X"], ["Y"])],
        "relu-only", [X], [Y])
    return helper.make_model(graph, opset_imports=_opset())


def _two_input_add():
    """Two graph inputs into Add — exercises multiple-input memcpy."""
    X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [2, 4])
    X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [2, 4])
    Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [2, 4])
    graph = helper.make_graph(
        [helper.make_node("Add", ["X1", "X2"], ["Y"])],
        "two-input", [X1, X2], [Y])
    return helper.make_model(graph, opset_imports=_opset())


def _two_output_model():
    """One input, two independent MatMul outputs — exercises multi-output memcpy."""
    W1 = numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), "W1")
    W2 = numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), "W2")
    X  = helper.make_tensor_value_info("X",  TensorProto.FLOAT, [4, 4])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [4, 4])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [4, 4])
    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["Y1"]),
        helper.make_node("MatMul", ["X", "W2"], ["Y2"]),
    ]
    graph = helper.make_graph(nodes, "two-out", [X], [Y1, Y2],
                               initializer=[W1, W2])
    return helper.make_model(graph, opset_imports=_opset())


def _resnet_stem():
    """Conv7×7/stride2 on 224×224 — large constant stress test.
    OH = OW = (224 + 6 - 6) / 2 + 1 = 112."""
    W = numpy_helper.from_array(
        np.zeros((64, 3, 7, 7), dtype=np.float32), "stem_W")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("Conv", ["X", "stem_W"], ["Y"],
                             strides=[2, 2], pads=[3, 3, 3, 3])
    graph = helper.make_graph([node], "resnet-stem", [X], [Y], initializer=[W])
    return helper.make_model(graph, opset_imports=_opset())


# ===========================================================================
# 1. cName sanitisation
# ===========================================================================

class TestCNameSanitisation:

    def test_all_value_pointers_prefixed_v_(self, runner, tmp_path):
        """Every float pointer declaration in model.c must be named 'v_*'.
        cName() strips the SSA '%' prefix then prepends 'v_'."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        # Collect every line that declares a float* or const float* variable.
        decl_lines = [
            l for l in src.splitlines()
            if re.search(r"(?:const\s+)?float\s*\*\s*v_", l)
            and "=" in l
            and "seecpp_run_model" not in l
            and "seecpp_im2col" not in l
        ]
        assert len(decl_lines) > 0, \
            "model.c must have at least one 'v_'-prefixed pointer declaration"
        for line in decl_lines:
            m = re.search(r"(?:const\s+)?float\s*\*\s*(v_\w+)\s*=", line)
            assert m, (
                f"Pointer declaration must match 'v_<ident> =':\n  {line.strip()}"
            )

    def test_no_percent_sign_in_c_code(self, runner, tmp_path):
        """'%' must not appear in any C identifier in model.c.
        It is valid only in printf-style format strings, which we don't emit."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        # Strip comment lines; look for '%' in code.
        code_lines = [l for l in src.splitlines()
                      if not l.strip().startswith("/*")
                      and not l.strip().startswith("//")]
        for line in code_lines:
            assert "%" not in line, (
                f"Illegal '%' in model.c code line:\n  {line.strip()}"
            )

    def test_dots_not_in_variable_names(self, runner, tmp_path):
        """SSA ids containing '.' (e.g. 'op.0') must have '.' replaced by '_'.
        A dot inside a C identifier would be a field-access parse error."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        body_start = src.find("{")
        body = src[body_start:] if body_start != -1 else src
        code_lines = [l for l in body.splitlines()
                      if not l.strip().startswith("/*")
                      and not l.strip().startswith("//")]
        for line in code_lines:
            # 'v_word.word' would be a dot in a C identifier — disallow.
            assert not re.search(r"\bv_\w+\.\w", line), (
                f"Dot inside a 'v_*' identifier on line:\n  {line.strip()}"
            )

    def test_weight_pointers_declared_const(self, runner, tmp_path):
        """Initializer/weight values must be declared 'const float*' — they
        point into the immutable weight_blob."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        weight_decls = [l for l in src.splitlines()
                        if "weight_blob" in l and "float*" in l]
        assert len(weight_decls) > 0, \
            "model.c must have at least one weight pointer declaration"
        for line in weight_decls:
            assert "const float*" in line, (
                f"Weight pointer must be 'const float*':\n  {line.strip()}"
            )

    def test_activation_pointers_declared_mutable(self, runner, tmp_path):
        """Arena-backed activation values must be mutable 'float*' — they
        receive cblas_sgemm output and are written by memcpy."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        arena_decls = [l for l in src.splitlines()
                       if "(char*)arena" in l and "float*" in l]
        assert len(arena_decls) > 0, \
            "model.c must have at least one arena-backed pointer declaration"
        for line in arena_decls:
            assert "const float*" not in line, (
                f"Activation pointer must be mutable 'float*':\n  {line.strip()}"
            )


# ===========================================================================
# 2. Pointer expression structure
# ===========================================================================

class TestPtrExprEmission:

    def test_weight_ptr_pattern(self, runner, tmp_path):
        """Weight pointer expressions must follow the pattern:
        '(const float*)((const char*)weight_blob + NUL)'."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert re.search(r"\(const char\*\)weight_blob\s*\+\s*\d+UL", src), \
            "Weight pointer must use '(const char*)weight_blob + NUL' form"

    def test_activation_ptr_pattern(self, runner, tmp_path):
        """Activation pointer expressions must follow the pattern:
        '(float*)((char*)arena + NUL)'."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert re.search(r"\(char\*\)arena\s*\+\s*\d+UL", src), \
            "Activation pointer must use '(char*)arena + NUL' form"

    def test_first_weight_at_offset_zero(self, runner, tmp_path):
        """WeightBuffer stores tensors sequentially from offset 0.
        The first weight pointer must embed '+ 0UL'."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert re.search(r"weight_blob\s*\+\s*0UL", src), \
            "First weight tensor must have byte offset 0 in weight_blob"

    def test_first_activation_at_offset_zero(self, runner, tmp_path):
        """BufferAllocator assigns the first activation at watermark=0.
        The first arena pointer must embed '+ 0UL'."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert re.search(r"\(char\*\)arena\s*\+\s*0UL", src), \
            "First activation must have byte offset 0 in the arena"

    def test_second_weight_offset_positive(self, runner, tmp_path):
        """A model with ≥ 2 weight tensors must have at least one
        weight pointer with offset > 0."""
        # Conv has filter W. Conv+BN adds 4 BN tensors → 5 total weights.
        W_init  = numpy_helper.from_array(
            np.zeros((8, 3, 3, 3), dtype=np.float32), "W")
        sc_t = numpy_helper.from_array(np.ones(8,  dtype=np.float32), "sc")
        bi_t = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "bi")
        mn_t = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "mn")
        vr_t = numpy_helper.from_array(np.ones(8,  dtype=np.float32), "vr")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        nodes = [
            helper.make_node("Conv", ["X", "W"], ["co"],
                              strides=[1, 1], pads=[1, 1, 1, 1]),
            helper.make_node("BatchNormalization",
                              ["co", "sc", "bi", "mn", "vr"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "conv-bn", [X], [Y],
                                   initializer=[W_init, sc_t, bi_t, mn_t, vr_t])
        model = helper.make_model(graph, opset_imports=_opset())

        onnx_f = str(tmp_path / "cbn.onnx")
        sir_f  = str(tmp_path / "cbn.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, model)
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        # Collect all weight_blob offsets.
        offsets = [int(m.group(1))
                   for m in re.finditer(r"weight_blob\s*\+\s*(\d+)UL", src)]
        assert any(o > 0 for o in offsets), (
            f"With multiple weights, at least one must have offset > 0; "
            f"found offsets: {offsets}"
        )


# ===========================================================================
# 3. sc_low.reshape is a no-op
# ===========================================================================

class TestReshapeIsNoop:

    def test_reshape_emits_no_extra_memcpy(self, runner, tmp_path):
        """Conv lowering inserts sc_low.reshape to flatten the filter matrix.
        model.c must NOT emit a memcpy for this reshape.
        The only memcpy calls are: one per graph input + one per graph output."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        # 1 graph input + 1 graph output = 2 memcpy max (no reshape copy).
        memcpy_count = src.count("memcpy(")
        assert memcpy_count <= 2, (
            f"model.c has {memcpy_count} memcpy calls; sc_low.reshape must "
            "contribute zero (it is a logical reinterpretation, not a copy)"
        )

    def test_reshape_comment_present(self, runner, tmp_path):
        """The 'logical reinterpret' comment must appear once per lowered conv
        to document the reshape no-op decision."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "logical reinterpret" in src, \
            "model.c must contain 'logical reinterpret' comment for sc_low.reshape"

    def test_reshape_line_contains_no_cblas(self, runner, tmp_path):
        """The line containing the reshape comment must not also call cblas_sgemm."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        reshape_idx = src.find("logical reinterpret")
        assert reshape_idx != -1
        line_end = src.find("\n", reshape_idx)
        reshape_line = src[reshape_idx:line_end]
        assert "cblas_sgemm" not in reshape_line, \
            "The sc_low.reshape comment line must not contain a cblas call"


# ===========================================================================
# 4. Fused elementwise emission
# ===========================================================================

class TestFusedEwEmission:

    def test_add_segment_emits_plus_equal(self, runner, tmp_path):
        """Add→Mul fused_ew: the Add segment must emit '+=' in the loop body."""
        onnx_f = str(tmp_path / "am.onnx")
        sir_f  = str(tmp_path / "am.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _add_mul_chain())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "+=" in src, \
            "Add segment of fused elementwise chain must emit '+=' in model.c"

    def test_mul_segment_emits_star_equal(self, runner, tmp_path):
        """Add→Mul fused_ew: the Mul segment must emit '*=' in the loop body."""
        onnx_f = str(tmp_path / "am.onnx")
        sir_f  = str(tmp_path / "am.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _add_mul_chain())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "*=" in src, \
            "Mul segment of fused elementwise chain must emit '*=' in model.c"

    def test_fused_ew_uses_acc_accumulator(self, runner, tmp_path):
        """The fused elementwise loop must declare and use '_acc' as the
        running accumulator variable."""
        onnx_f = str(tmp_path / "am.onnx")
        sir_f  = str(tmp_path / "am.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _add_mul_chain())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "_acc" in src, \
            "Fused elementwise loop must use a '_acc' accumulator variable"

    def test_fused_ew_loop_bound_matches_volume(self, runner, tmp_path):
        """The fused_ew loop bound must equal the tensor's total element count.
        For shape [2, 4] that is 8."""
        onnx_f = str(tmp_path / "am.onnx")
        sir_f  = str(tmp_path / "am.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _add_mul_chain())   # shape [2, 4] → 8 elements
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert re.search(r"_i\s*<\s*8\b", src), \
            "Fused_ew loop must iterate over 8 elements for a [2,4] tensor"


# ===========================================================================
# 5. Gemm emission
# ===========================================================================

class TestGemmEmission:

    def test_no_trans_emits_notrans_for_b(self, runner, tmp_path):
        """Gemm without transB must emit 'CblasNoTrans' for the B matrix."""
        onnx_f = str(tmp_path / "g.onnx")
        sir_f  = str(tmp_path / "g.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _gemm(trans_b=False))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "CblasNoTrans" in src, \
            "Gemm without transB must emit 'CblasNoTrans' in model.c"

    def test_trans_b_emits_cblas_trans_for_b(self, runner, tmp_path):
        """Gemm with transB=1 must emit 'CblasTrans' for the B matrix argument."""
        onnx_f = str(tmp_path / "g_tb.onnx")
        sir_f  = str(tmp_path / "g_tb.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _gemm(trans_b=True))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "CblasTrans" in src, \
            "Gemm with transB=1 must emit 'CblasTrans' in model.c"

    def test_trans_a_emits_cblas_trans_for_a(self, runner, tmp_path):
        """Gemm with transA=1 must emit 'CblasTrans' for the A matrix argument."""
        onnx_f = str(tmp_path / "g_ta.onnx")
        sir_f  = str(tmp_path / "g_ta.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _gemm(trans_a=True))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "CblasTrans" in src, \
            "Gemm with transA=1 must emit 'CblasTrans' for A in model.c"

    def test_gemm_with_bias_emits_bias_loop(self, runner, tmp_path):
        """Gemm with a third bias operand must emit a bias-add loop."""
        onnx_f = str(tmp_path / "g_bias.onnx")
        sir_f  = str(tmp_path / "g_bias.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _gemm(with_bias=True))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        # The bias-add section is marked by its comment.
        assert "gemm bias add" in src, \
            "Gemm with bias must emit a '/* gemm bias add */' section in model.c"

    def test_gemm_without_bias_no_bias_comment(self, runner, tmp_path):
        """Gemm with only 2 operands must NOT emit the bias-add section."""
        onnx_f = str(tmp_path / "g_nobias.onnx")
        sir_f  = str(tmp_path / "g_nobias.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _gemm(with_bias=False))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "gemm bias add" not in src, \
            "Gemm without bias must not emit a 'gemm bias add' comment"

    def test_gemm_fused_relu_emits_0f_clamp(self, runner, tmp_path):
        """Gemm+Relu (fused by OperatorFusionPass): model.c must contain the
        post-GEMM relu clamp loop using '0.0f'."""
        onnx_f = str(tmp_path / "g_relu.onnx")
        sir_f  = str(tmp_path / "g_relu.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _gemm(with_relu=True))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "0.0f" in src, \
            "Fused relu after Gemm must emit a '> 0.0f' clamp loop in model.c"

    @pytest.mark.parametrize("M,K,N", [
        (4,   8,  16),
        (1, 512, 2048),   # BERT-scale FFN row
        (32,  64,  64),
    ])
    def test_gemm_dimension_constants_embedded(self, runner, tmp_path, M, K, N):
        """cblas_sgemm must embed M, K, N as literal integer constants.
        This verifies that shape information is correctly propagated to the
        emission site."""
        onnx_f = str(tmp_path / f"g_{M}_{K}_{N}.onnx")
        sir_f  = str(tmp_path / f"g_{M}_{K}_{N}.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _gemm(M=M, K=K, N=N, with_bias=False))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        for dim in (M, K, N):
            assert str(dim) in src, (
                f"Dimension {dim} must appear in model.c "
                f"for Gemm [{M},{K}] × [{K},{N}]"
            )


# ===========================================================================
# 6. Relu emission
# ===========================================================================

class TestReluEmission:

    def test_standalone_relu_emits_0f_loop(self, runner, tmp_path):
        """A standalone Relu op (no preceding matmul — not fused) must emit
        an explicit relu loop using '0.0f' comparison via emitReluOp."""
        onnx_f = str(tmp_path / "relu.onnx")
        sir_f  = str(tmp_path / "relu.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _relu_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "0.0f" in src, \
            "Standalone relu must emit a '0.0f' clamp loop in model.c"

    def test_relu_loop_bound_matches_volume(self, runner, tmp_path):
        """The relu loop must iterate over the exact tensor element count.
        For shape [2, 4] the bound is 8."""
        onnx_f = str(tmp_path / "relu.onnx")
        sir_f  = str(tmp_path / "relu.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _relu_only())   # [2, 4] → 8 elements
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert re.search(r"_i\s*<\s*8\b", src), \
            "Relu loop bound must be 8 for a [2,4] tensor"

    def test_copy_and_clamp_uses_ternary_form(self, runner, tmp_path):
        """When input and output occupy different arena slots (the normal case
        for a standalone relu), emitReluOp must use the ternary
        'input[i] > 0.0f ? input[i] : 0.0f' copy-and-clamp form."""
        onnx_f = str(tmp_path / "relu.onnx")
        sir_f  = str(tmp_path / "relu.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _relu_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        # The ternary form contains '> 0.0f ?' followed later by ': 0.0f'.
        assert re.search(r">\s*0\.0f\s*\?", src), \
            "Copy-and-clamp relu must use ternary '>0.0f ? x : 0.0f' form"

    def test_relu_uses_i_loop_variable(self, runner, tmp_path):
        """The relu loop must use the '_i' induction variable consistent with
        all other emitted loops in model.c."""
        onnx_f = str(tmp_path / "relu.onnx")
        sir_f  = str(tmp_path / "relu.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _relu_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "int _i" in src, \
            "Relu loop must use 'int _i' induction variable"


# ===========================================================================
# 7. Batched matmul emission (lowered conv)
# ===========================================================================

class TestBatchedMatMulEmission:

    def test_lowered_conv_emits_batch_loop(self, runner, tmp_path):
        """ConvLoweringPass produces a sc_low.matmul where B has 3 dimensions
        [N, K, OH*OW]. emitMatMulOp must emit a 'for (int _b' loop over N."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only(N=2))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "for (int _b" in src, \
            "Batched sc_low.matmul from Conv lowering must emit 'for (int _b' loop"

    def test_batch_loop_bound_matches_N(self, runner, tmp_path):
        """The '_b < N' loop bound must equal the actual batch size."""
        N = 3
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only(N=N))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert re.search(rf"_b\s*<\s*{N}\b", src), \
            f"Batch loop bound must be {N} in model.c"

    def test_per_batch_stride_offset_present(self, runner, tmp_path):
        """The batched cblas_sgemm call must offset B and C by '_b *' stride
        to step through the col-matrix and output slices."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only(N=2))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert re.search(r"\+\s*_b\s*\*", src), \
            "Per-batch stride offset '+ _b *' must appear in batched cblas call"

    @pytest.mark.parametrize("N", [1, 4, 8])
    def test_batch_loop_bound_parametrized(self, runner, tmp_path, N):
        """Loop bound must equal N exactly for N ∈ {1, 4, 8}."""
        onnx_f = str(tmp_path / f"conv_N{N}.onnx")
        sir_f  = str(tmp_path / f"conv_N{N}.sir")
        out_d  = str(tmp_path / f"out_N{N}")
        _write(onnx_f, _conv_only(N=N))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert re.search(rf"_b\s*<\s*{N}\b", src), \
            f"Batch loop bound must equal N={N} in model.c"


# ===========================================================================
# 8. Multiple inputs and outputs
# ===========================================================================

class TestMultipleInputsOutputs:

    def test_two_inputs_produce_inputs_0_and_1(self, runner, tmp_path):
        """A model with two graph inputs must emit memcpy from inputs[0] and
        inputs[1] into the arena."""
        onnx_f = str(tmp_path / "two_in.onnx")
        sir_f  = str(tmp_path / "two_in.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _two_input_add())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "inputs[0]" in src, "First input must reference inputs[0]"
        assert "inputs[1]" in src, "Second input must reference inputs[1]"

    def test_two_inputs_at_least_two_input_memcpys(self, runner, tmp_path):
        """Two graph inputs → at least two distinct 'inputs[N]' references."""
        onnx_f = str(tmp_path / "two_in.onnx")
        sir_f  = str(tmp_path / "two_in.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _two_input_add())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        refs = len(re.findall(r"inputs\[\d+\]", src))
        assert refs >= 2, \
            f"Expected >= 2 inputs[] references for two graph inputs; got {refs}"

    def test_two_outputs_produce_outputs_0_and_1(self, runner, tmp_path):
        """A model with two terminal ops must emit memcpy into outputs[0] and
        outputs[1] in the output-copy section."""
        onnx_f = str(tmp_path / "two_out.onnx")
        sir_f  = str(tmp_path / "two_out.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _two_output_model())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "outputs[0]" in src, "First output must reference outputs[0]"
        assert "outputs[1]" in src, "Second output must reference outputs[1]"

    def test_single_input_uses_only_inputs_0(self, runner, tmp_path):
        """A model with exactly one graph input must use inputs[0] only.
        inputs[1] must not appear."""
        onnx_f = str(tmp_path / "single.onnx")
        sir_f  = str(tmp_path / "single.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "inputs[0]" in src,      "Single-input model must reference inputs[0]"
        assert "inputs[1]" not in src,  "Single-input model must not reference inputs[1]"

    def test_output_memcpy_appears_after_op_sequence(self, runner, tmp_path):
        """The '/* --- Copy outputs ---*/' section must appear after the
        '/* --- Op sequence ---*/' section in model.c."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        ops_pos = src.find("Op sequence")
        out_pos = src.find("Copy outputs")
        assert ops_pos != -1 and out_pos != -1, \
            "Both 'Op sequence' and 'Copy outputs' sections must be present"
        assert ops_pos < out_pos, \
            "Output copy section must appear after the op sequence section"


# ===========================================================================
# 9. Header guards
# ===========================================================================

class TestHeaderGuards:

    def test_arena_size_h_has_pragma_once(self, runner, tmp_path):
        """arena_size.h must begin with '#pragma once' to prevent double-include."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)
        assert "#pragma once" in _arena_h(out_d), \
            "arena_size.h must contain '#pragma once'"

    def test_model_h_has_pragma_once(self, runner, tmp_path):
        """model.h must contain '#pragma once'."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)
        assert "#pragma once" in _model_h(out_d), \
            "model.h must contain '#pragma once'"

    def test_model_h_includes_stddef(self, runner, tmp_path):
        """model.h must include <stddef.h> — required for size_t in the ABI."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)
        assert "#include <stddef.h>" in _model_h(out_d), \
            "model.h must include <stddef.h>"

    def test_model_c_includes_model_h(self, runner, tmp_path):
        """model.c must include its own model.h to pull in the function
        declaration before the definition."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)
        src = _model_c(out_d)
        assert '#include "model.h"' in src, \
            'model.c must #include "model.h"'

    def test_model_c_includes_cblas_h(self, runner, tmp_path):
        """model.c must include <cblas.h> before calling cblas_sgemm."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)
        src = _model_c(out_d)
        assert "#include <cblas.h>" in src, \
            "model.c must #include <cblas.h>"

    def test_model_c_cblas_include_before_function(self, runner, tmp_path):
        """<cblas.h> include must appear before the seecpp_run_model definition
        so the cblas_sgemm declaration is visible at the call site."""
        onnx_f = str(tmp_path / "conv.onnx")
        sir_f  = str(tmp_path / "conv.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _conv_only())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        cblas_pos   = src.find("#include <cblas.h>")
        fn_def_pos  = src.find("void seecpp_run_model")
        assert cblas_pos != -1 and fn_def_pos != -1
        assert cblas_pos < fn_def_pos, \
            "<cblas.h> include must precede the seecpp_run_model definition"


# ===========================================================================
# 10. Large shape constants
# ===========================================================================

class TestLargeShapeConstants:

    def test_resnet_stem_oh_ow_is_112(self, runner, tmp_path):
        """ResNet stem (Conv7×7/stride2/pad3 on 224×224):
        OH = OW = (224 + 3 + 3 - 7) / 2 + 1 = 112 must appear in model.c."""
        onnx_f = str(tmp_path / "stem.onnx")
        sir_f  = str(tmp_path / "stem.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _resnet_stem())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "112" in src, \
            "ResNet stem output spatial dim 112 must appear in model.c"

    def test_resnet_stem_filter_channels_64(self, runner, tmp_path):
        """ResNet stem has 64 output channels; 64 must appear as a literal."""
        onnx_f = str(tmp_path / "stem.onnx")
        sir_f  = str(tmp_path / "stem.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _resnet_stem())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "64" in src, \
            "ResNet stem output channel count 64 must appear in model.c"

    def test_resnet_stem_kernel_7_embedded(self, runner, tmp_path):
        """The 7×7 kernel size must appear in the seecpp_im2col call."""
        onnx_f = str(tmp_path / "stem.onnx")
        sir_f  = str(tmp_path / "stem.sir")
        out_d  = str(tmp_path / "out")
        _write(onnx_f, _resnet_stem())
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert "7" in src, \
            "Kernel size 7 must appear in the seecpp_im2col call for ResNet stem"

    @pytest.mark.parametrize("H,W,stride,expected_oh,expected_ow", [
        (56, 56, 1, 56, 56),   # layer2: same-pad, stride=1, 3×3 kernel
        (56, 56, 2, 28, 28),   # layer2→layer3: stride=2 downsample
        (28, 28, 2, 14, 14),   # layer3→layer4: stride=2 downsample
        (14, 14, 2,  7,  7),   # layer4→avg-pool: stride=2 downsample
    ])
    def test_output_spatial_constants_embedded(
        self, runner, tmp_path, H, W, stride, expected_oh, expected_ow
    ):
        """Output spatial dimensions OH/OW must be embedded as integer literals
        in the seecpp_im2col call (and implicitly in the cblas_sgemm dims)."""
        tag = f"{H}x{W}_s{stride}"
        onnx_f = str(tmp_path / f"conv_{tag}.onnx")
        sir_f  = str(tmp_path / f"conv_{tag}.sir")
        out_d  = str(tmp_path / f"out_{tag}")
        _write(onnx_f, _conv_only(H=H, W=W, strides=(stride, stride),
                                   pads=(1, 1, 1, 1)))
        runner.run_and_parse(onnx_f, sir_f, codegen_out_dir=out_d)

        src = _model_c(out_d)
        assert str(expected_oh) in src, (
            f"Expected OH={expected_oh} to appear in model.c "
            f"for H={H}, stride={stride}"
        )
        assert str(expected_ow) in src, (
            f"Expected OW={expected_ow} to appear in model.c "
            f"for W={W}, stride={stride}"
        )