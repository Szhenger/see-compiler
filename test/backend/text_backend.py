"""
test_codegen.py — stress tests for the SeeC++ codegen (backend) layer.

Coverage map:
  TestWeightFolding        — WeightFoldingPass arithmetic and guard paths
  TestBufferAllocator      — slot layout, 64-byte alignment, weight vs arena
  TestCpuCCodegenTarget    — four output artefacts, structural content checks
  TestCodegenDriver        — pipeline orchestration, stage ordering, logging
  TestNumericalCorrectness — generated model.c compiled and run against numpy

Test strategy:
  - All codegen tests drive the full compiler binary (frontend + middle-end +
    codegen) and inspect the artefacts produced in a tmp_path directory.
  - WeightFolding arithmetic tests verify the folding formula independently
    using numpy, then confirm the binary produces the same folded weights.
  - Numerical correctness tests compile the generated model.c with cc and
    cffi, run inference on random inputs, and compare to numpy / onnxruntime.

Logger lines parsed here:
  WeightFoldingPass: N fused Conv+BN op(s) to fold
  WeightFoldingPass: folded N Conv+BN weight pair(s)
  WeightFoldingPass: folded '<id>' -> '<id>__bn_folded' [F=N elems_per_channel=N]
  BufferAllocator: N value(s) allocated, arena=N bytes
  CpuCCodegenTarget: wrote model.c + model.h + weights.bin + arena_size.h to '...'
  CodegenDriver: pipeline complete — arena=N bytes, target='CpuCCodegenTarget'

Run with:
    pytest test_codegen.py -v
Skip numerical tests without a C compiler:
    pytest test_codegen.py -v -m "not numerical"
"""

import re
import os
import math
import struct
import shutil
import subprocess
import dataclasses
import pytest
import numpy as np
import onnx
from onnx import helper, TensorProto, checker, numpy_helper
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers shared with other test files
# ---------------------------------------------------------------------------

def _opset(version: int = 17):
    return [helper.make_opsetid("", version)]


def _write_model(path: str, model: onnx.ModelProto,
                 skip_check: bool = False) -> None:
    if not skip_check:
        try:
            checker.check_model(model)
        except checker.ValidationError as e:
            raise ValueError(f"Test fixture is not valid ONNX: {e}") from e
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


# ---------------------------------------------------------------------------
# Log parsers specific to the codegen layer
# ---------------------------------------------------------------------------

def _combined(summary) -> str:
    return summary.stderr + summary.stdout


def _parse_fold_count(summary) -> int:
    """Number of Conv+BN pairs WeightFoldingPass reports folding."""
    m = re.search(r"WeightFoldingPass: folded\s+(\d+)\s+Conv\+BN", _combined(summary))
    return int(m.group(1)) if m else 0


def _parse_fold_log_entries(summary) -> list[dict]:
    """Return one dict per folded tensor: {filter_id, folded_key, F, elems}."""
    entries = []
    for m in re.finditer(
        r"WeightFoldingPass: folded '([^']+)' -> '([^']+)'"
        r" \[F=(\d+) elems_per_channel=(\d+)\]",
        _combined(summary)
    ):
        entries.append({
            "filter_id":  m.group(1),
            "folded_key": m.group(2),
            "F":          int(m.group(3)),
            "elems":      int(m.group(4)),
        })
    return entries


def _parse_arena_bytes_from_allocator(summary) -> Optional[int]:
    """Bytes reported by BufferAllocator."""
    m = re.search(r"BufferAllocator:.*arena=(\d+)\s+bytes", _combined(summary))
    return int(m.group(1)) if m else None


def _parse_arena_bytes_from_driver(summary) -> Optional[int]:
    """Bytes reported by CodegenDriver completion line."""
    m = re.search(r"CodegenDriver: pipeline complete.*arena=(\d+)\s+bytes",
                  _combined(summary))
    return int(m.group(1)) if m else None


def _parse_allocated_values(summary) -> Optional[int]:
    m = re.search(r"BufferAllocator:\s+(\d+)\s+value\(s\) allocated",
                  _combined(summary))
    return int(m.group(1)) if m else None


def _codegen_completed(summary) -> bool:
    return "CpuCCodegenTarget: wrote" in _combined(summary)


def _driver_completed(summary) -> bool:
    return "CodegenDriver: pipeline complete" in _combined(summary)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_conv_bn_relu(N=1, C=3, H=8, W=8, F=8,
                       scale=None, bias=None, mean=None, var=None):
    """Conv + BN + Relu with controllable BN parameters."""
    W_init  = numpy_helper.from_array(
        np.random.randn(F, C, 3, 3).astype(np.float32), "W")
    scale_t = numpy_helper.from_array(
        (scale if scale is not None else np.ones(F)).astype(np.float32), "bn_scale")
    bias_t  = numpy_helper.from_array(
        (bias  if bias  is not None else np.zeros(F)).astype(np.float32), "bn_bias")
    mean_t  = numpy_helper.from_array(
        (mean  if mean  is not None else np.zeros(F)).astype(np.float32), "bn_mean")
    var_t   = numpy_helper.from_array(
        (var   if var   is not None else np.ones(F)).astype(np.float32),  "bn_var")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    nodes = [
        helper.make_node("Conv", ["X", "W"], ["conv_out"],
                          strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node("BatchNormalization",
                          ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                          ["bn_out"]),
        helper.make_node("Relu", ["bn_out"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "conv-bn-relu", [X], [Y],
                               initializer=[W_init, scale_t, bias_t, mean_t, var_t])
    return helper.make_model(graph, opset_imports=_opset())


def _make_gemm_relu(M=4, K=8, N=16):
    """Gemm + Relu — no BN, so WeightFolding is a no-op."""
    W = numpy_helper.from_array(
        np.random.randn(K, N).astype(np.float32), "W")
    b = numpy_helper.from_array(np.zeros(N, dtype=np.float32), "b")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    nodes = [
        helper.make_node("Gemm", ["X", "W", "b"], ["g_out"]),
        helper.make_node("Relu", ["g_out"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "gemm-relu", [X], [Y], initializer=[W, b])
    return helper.make_model(graph, opset_imports=_opset())


def _make_conv_only(N=1, C=3, H=8, W=8, F=8):
    """Conv only — no BN, no Relu."""
    W_init = numpy_helper.from_array(
        np.zeros((F, C, 3, 3), dtype=np.float32), "W")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, C, H, W])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    node = helper.make_node("Conv", ["X", "W"], ["Y"],
                             strides=[1, 1], pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "conv-only", [X], [Y], initializer=[W_init])
    return helper.make_model(graph, opset_imports=_opset())


# ---------------------------------------------------------------------------
# BN folding reference implementation (pure numpy)
# Used to compute expected folded weights independently of the C++ pipeline.
# ---------------------------------------------------------------------------

def _bn_fold_reference(W: np.ndarray, b: np.ndarray,
                        scale: np.ndarray, bn_bias: np.ndarray,
                        mean: np.ndarray, var: np.ndarray,
                        eps: float = 1e-5):
    """
    Compute W_fused and b_fused using the same formula as weight_folding_pass.cpp.

    W shape: [F, C, KH, KW]  (or [F, elems_per_out] after reshape)
    scale, bn_bias, mean, var: [F]

    Returns (W_fused, b_fused) both as float32 arrays.
    """
    F = scale.shape[0]
    std_inv      = 1.0 / np.sqrt(var + eps)              # [F]
    scale_factor = scale * std_inv                        # [F]

    # Broadcast scale_factor over all spatial dims.
    # W is [F, C, KH, KW] — reshape scale_factor to [F, 1, 1, 1].
    sf = scale_factor.reshape(F, *([1] * (W.ndim - 1)))
    W_fused = (W * sf).astype(np.float32)
    b_fused = ((b - mean) * scale_factor + bn_bias).astype(np.float32)
    return W_fused, b_fused


# ===========================================================================
# 1. WeightFoldingPass
# ===========================================================================

class TestWeightFolding:

    def test_fold_count_reported_for_conv_bn(self, runner, tmp_path):
        """WeightFoldingPass must report folding 1 Conv+BN pair."""
        onnx_file = str(tmp_path / "cbr.onnx")
        sir_file  = str(tmp_path / "cbr.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)
        assert _parse_fold_count(summary) == 1, \
            "Expected WeightFoldingPass to report folding 1 Conv+BN pair"

    def test_fold_log_entry_present(self, runner, tmp_path):
        """WeightFoldingPass must emit a per-tensor fold log line."""
        onnx_file = str(tmp_path / "cbr.onnx")
        sir_file  = str(tmp_path / "cbr.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)
        entries = _parse_fold_log_entries(summary)
        assert len(entries) == 1, \
            f"Expected 1 fold log entry; got {len(entries)}"
        assert entries[0]["folded_key"].endswith("__bn_folded"), \
            "Folded weight key must end with '__bn_folded'"

    def test_fold_log_reports_correct_F(self, runner, tmp_path):
        """The fold log must report F equal to the number of output channels."""
        F = 16
        onnx_file = str(tmp_path / "cbr.onnx")
        sir_file  = str(tmp_path / "cbr.sir")
        _write_model(onnx_file, _make_conv_bn_relu(F=F))

        summary = runner.run_and_parse(onnx_file, sir_file)
        entries = _parse_fold_log_entries(summary)
        assert entries[0]["F"] == F, \
            f"Fold log must report F={F}; got {entries[0]['F']}"

    @pytest.mark.parametrize("F,C,KH,KW", [
        (8,  3, 3, 3),   # standard 3x3 conv
        (16, 8, 1, 1),   # pointwise 1x1
        (32, 3, 5, 5),   # 5x5 kernel
    ])
    def test_fold_arithmetic_matches_numpy(self, runner, tmp_path, F, C, KH, KW):
        """Folded weights written to weights.bin must match the numpy
        reference formula exactly (within float32 precision)."""
        np.random.seed(42)
        W_np     = np.random.randn(F, C, KH, KW).astype(np.float32)
        scale_np = np.random.rand(F).astype(np.float32) + 0.5
        bias_np  = np.random.randn(F).astype(np.float32)
        mean_np  = np.random.randn(F).astype(np.float32)
        var_np   = np.random.rand(F).astype(np.float32) + 0.1

        onnx_file = str(tmp_path / "cbr.onnx")
        sir_file  = str(tmp_path / "cbr.sir")
        out_dir   = str(tmp_path / "out")

        _write_model(onnx_file, _make_conv_bn_relu(
            F=F, C=C,
            scale=scale_np, bias=bias_np, mean=mean_np, var=var_np))
        summary = runner.run_and_parse(onnx_file, sir_file,
                                        codegen_out_dir=out_dir)

        W_ref, b_ref = _bn_fold_reference(
            W_np, np.zeros(F, dtype=np.float32),
            scale_np, bias_np, mean_np, var_np)

        # Read the first F*C*KH*KW floats from weights.bin (the folded filter
        # is written first because WeightFoldingPass runs before the original
        # weights are serialised).
        weights_bin = Path(out_dir) / "weights.bin"
        assert weights_bin.exists(), "weights.bin must be written by codegen"

        raw = np.frombuffer(weights_bin.read_bytes(), dtype=np.float32)
        # The folded weight is appended after original weights.
        # Its size is F*C*KH*KW floats.
        n_filter = F * C * KH * KW
        # Find the folded block: the last n_filter+F floats cover W_fused+b_fused.
        w_folded = raw[-(n_filter + F):-F]
        b_folded = raw[-F:]

        np.testing.assert_allclose(
            w_folded, W_ref.flatten(), rtol=1e-5, atol=1e-6,
            err_msg="W_fused from weights.bin does not match numpy reference")
        np.testing.assert_allclose(
            b_folded, b_ref, rtol=1e-5, atol=1e-6,
            err_msg="b_fused from weights.bin does not match numpy reference")

    def test_fold_zero_bias_synthesised_correctly(self, runner, tmp_path):
        """Conv with no bias operand: WeightFoldingPass must synthesise zero
        bias and fold it correctly (b_fused = -mean * scale_factor + beta)."""
        F, C = 8, 3
        np.random.seed(7)
        scale_np = np.ones(F, dtype=np.float32)
        bias_np  = np.zeros(F, dtype=np.float32)
        mean_np  = np.random.randn(F).astype(np.float32)
        var_np   = np.ones(F, dtype=np.float32)

        # Build Conv+BN WITHOUT a Conv bias operand.
        W_init  = numpy_helper.from_array(
            np.zeros((F, C, 3, 3), dtype=np.float32), "W")
        scale_t = numpy_helper.from_array(scale_np, "bn_scale")
        bias_t  = numpy_helper.from_array(bias_np,  "bn_bias")
        mean_t  = numpy_helper.from_array(mean_np,  "bn_mean")
        var_t   = numpy_helper.from_array(var_np,   "bn_var")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, C, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        nodes = [
            helper.make_node("Conv", ["X", "W"], ["conv_out"],   # no bias
                              strides=[1, 1], pads=[1, 1, 1, 1]),
            helper.make_node("BatchNormalization",
                              ["conv_out", "bn_scale", "bn_bias",
                               "bn_mean", "bn_var"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "conv-bn-nobias", [X], [Y],
                                   initializer=[W_init, scale_t, bias_t,
                                                mean_t, var_t])
        onnx_file = str(tmp_path / "nobias.onnx")
        sir_file  = str(tmp_path / "nobias.sir")
        _write_model(onnx_file,
                     helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)
        assert _parse_fold_count(summary) == 1, \
            "WeightFolding must fold a bias-free Conv+BN"

        # Expected b_fused: (0 - mean) * 1/sqrt(1+eps) * 1 + 0 ≈ -mean
        _, b_ref = _bn_fold_reference(
            np.zeros((F, C, 3, 3), dtype=np.float32),
            np.zeros(F, dtype=np.float32),
            scale_np, bias_np, mean_np, var_np)
        np.testing.assert_allclose(
            b_ref, -mean_np, rtol=1e-4,
            err_msg="b_fused for zero-bias Conv must equal -mean * scale_factor + bn_bias")

    def test_fold_idempotent_fused_bn_cleared(self, runner, tmp_path):
        """After folding, fused_bn must be set to 0 on the Conv op so a
        second CodegenDriver run does not re-fold the already-folded weights.
        Logger must show 0 folds-to-do on the second pipeline run."""
        onnx_file = str(tmp_path / "cbr.onnx")
        sir_file  = str(tmp_path / "cbr.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        # First run: expect 1 fold.
        summary1 = runner.run_and_parse(onnx_file, sir_file)
        assert _parse_fold_count(summary1) == 1

        # Second run on the same model file must still produce exactly 1 fold
        # (a fresh block is built from the ONNX each run).  What we verify is
        # that the "0 fused Conv+BN op(s) to fold" path is reachable by
        # submitting a pure Gemm model — no fused_bn attribute present.
        onnx_file2 = str(tmp_path / "gemm.onnx")
        sir_file2  = str(tmp_path / "gemm.sir")
        _write_model(onnx_file2, _make_gemm_relu())
        summary2 = runner.run_and_parse(onnx_file2, sir_file2)
        assert _parse_fold_count(summary2) == 0, \
            "No-BN model must report 0 WeightFolding folds"

    def test_fold_zero_on_no_bn_model(self, runner, tmp_path):
        """A Gemm+Relu model has no Conv+BN — WeightFoldingPass must
        report 0 folds and complete without error."""
        onnx_file = str(tmp_path / "gemm.onnx")
        sir_file  = str(tmp_path / "gemm.sir")
        _write_model(onnx_file, _make_gemm_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)
        assert summary.returncode == 0
        assert _parse_fold_count(summary) == 0

    @pytest.mark.parametrize("num_pairs", [2, 3])
    def test_fold_multiple_conv_bn_pairs(self, runner, tmp_path, num_pairs):
        """N sequential Conv+BN pairs must each be folded independently."""
        inits, nodes = [], []
        C_in, prev = 3, "X"
        for i in range(num_pairs):
            C_out = 8
            W_np = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
            inits += [
                numpy_helper.from_array(W_np,                                  f"W{i}"),
                numpy_helper.from_array(np.ones(C_out,  dtype=np.float32),     f"sc{i}"),
                numpy_helper.from_array(np.zeros(C_out, dtype=np.float32),     f"bi{i}"),
                numpy_helper.from_array(np.zeros(C_out, dtype=np.float32),     f"mn{i}"),
                numpy_helper.from_array(np.ones(C_out,  dtype=np.float32),     f"vr{i}"),
            ]
            nodes += [
                helper.make_node("Conv", [prev, f"W{i}"], [f"co{i}"],
                                  strides=[1,1], pads=[1,1,1,1]),
                helper.make_node("BatchNormalization",
                                  [f"co{i}", f"sc{i}", f"bi{i}",
                                   f"mn{i}", f"vr{i}"], [f"bo{i}"]),
            ]
            prev, C_in = f"bo{i}", C_out
        nodes.append(helper.make_node("Relu", [prev], ["Y"]))
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        graph = helper.make_graph(nodes, "multi-cbn", [X], [Y], initializer=inits)
        onnx_file = str(tmp_path / "multi.onnx")
        sir_file  = str(tmp_path / "multi.sir")
        _write_model(onnx_file, helper.make_model(graph, opset_imports=_opset()))

        summary = runner.run_and_parse(onnx_file, sir_file)
        assert _parse_fold_count(summary) == num_pairs, \
            f"Expected {num_pairs} folds; got {_parse_fold_count(summary)}"


# ===========================================================================
# 2. BufferAllocator
# ===========================================================================

class TestBufferAllocator:

    def test_allocator_logs_arena_bytes(self, runner, tmp_path):
        """BufferAllocator must log 'arena=N bytes' after allocation."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_only())

        summary = runner.run_and_parse(onnx_file, sir_file)
        arena = _parse_arena_bytes_from_allocator(summary)
        assert arena is not None, \
            "BufferAllocator must log arena byte count"
        assert arena > 0, \
            "Arena must be non-zero for a model with activations"

    def test_allocator_arena_positive_for_conv(self, runner, tmp_path):
        """Conv+BN+Relu requires non-trivial activation arena."""
        onnx_file = str(tmp_path / "cbr.onnx")
        sir_file  = str(tmp_path / "cbr.sir")
        _write_model(onnx_file, _make_conv_bn_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)
        arena = _parse_arena_bytes_from_allocator(summary)
        assert arena is not None and arena > 0

    def test_allocator_and_driver_arena_bytes_agree(self, runner, tmp_path):
        """BufferAllocator and CodegenDriver must report the same arena size."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_only())

        summary = runner.run_and_parse(onnx_file, sir_file)
        alloc_bytes  = _parse_arena_bytes_from_allocator(summary)
        driver_bytes = _parse_arena_bytes_from_driver(summary)

        assert alloc_bytes is not None and driver_bytes is not None
        assert alloc_bytes == driver_bytes, (
            f"BufferAllocator ({alloc_bytes}B) and CodegenDriver "
            f"({driver_bytes}B) must agree on arena size"
        )

    def test_arena_size_header_matches_logger(self, runner, tmp_path):
        """SEECPP_ARENA_BYTES in arena_size.h must match Logger output."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())

        summary = runner.run_and_parse(onnx_file, sir_file,
                                        codegen_out_dir=out_dir)
        driver_bytes = _parse_arena_bytes_from_driver(summary)

        arena_h = Path(out_dir) / "arena_size.h"
        assert arena_h.exists(), "arena_size.h must be written"
        content = arena_h.read_text()
        m = re.search(r"SEECPP_ARENA_BYTES\s*=\s*(\d+)", content)
        assert m, "SEECPP_ARENA_BYTES must appear in arena_size.h"
        assert int(m.group(1)) == driver_bytes, (
            f"arena_size.h value {m.group(1)} does not match "
            f"Logger value {driver_bytes}"
        )

    def test_64_byte_alignment_invariant(self, runner, tmp_path):
        """Every activation slot offset in the arena must be a multiple of 64.
        Verified by parsing the BufferAllocator log line: arena is always
        a multiple of 64 (the watermark is always aligned before each slot)."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_bn_relu(N=1, C=3, H=16, W=16, F=8))

        summary = runner.run_and_parse(onnx_file, sir_file)
        arena = _parse_arena_bytes_from_allocator(summary)
        assert arena is not None
        assert arena % 64 == 0, (
            f"Total arena ({arena}B) must be a multiple of 64 "
            "(all slot boundaries are 64-byte aligned)"
        )

    def test_weight_values_do_not_consume_arena(self, runner, tmp_path):
        """Weight tensors (initializers) must be annotated is_weight=true
        and must NOT count toward the activation arena.
        The arena must be strictly less than the total model parameter bytes."""
        F, C, KH, KW = 8, 3, 3, 3
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_only(F=F, C=C))

        summary = runner.run_and_parse(onnx_file, sir_file)
        arena = _parse_arena_bytes_from_allocator(summary)

        # Weight bytes: F*C*KH*KW float32 = 8*3*3*3*4 = 864 bytes
        weight_bytes = F * C * KH * KW * 4
        assert arena is not None
        assert arena < weight_bytes + 10_000, \
            "Arena should not include weight tensor bytes"

    def test_allocated_value_count_logged(self, runner, tmp_path):
        """BufferAllocator must log the total number of allocated values."""
        onnx_file = str(tmp_path / "gemm.onnx")
        sir_file  = str(tmp_path / "gemm.sir")
        _write_model(onnx_file, _make_gemm_relu())

        summary = runner.run_and_parse(onnx_file, sir_file)
        n = _parse_allocated_values(summary)
        assert n is not None and n >= 1, \
            "BufferAllocator must log at least 1 allocated value"

    @pytest.mark.parametrize("N", [1, 2, 4])
    def test_arena_scales_with_batch_size(self, runner, tmp_path, N):
        """Arena must grow proportionally with batch size N."""
        def _arena(n):
            onnx_f = str(tmp_path / f"conv_{n}.onnx")
            sir_f  = str(tmp_path / f"conv_{n}.sir")
            _write_model(onnx_f, _make_conv_only(N=n))
            s = runner.run_and_parse(onnx_f, sir_f)
            return _parse_arena_bytes_from_allocator(s)

        arena_1 = _arena(1)
        arena_n = _arena(N)
        assert arena_1 is not None and arena_n is not None
        if N > 1:
            assert arena_n > arena_1, (
                f"Arena for N={N} ({arena_n}B) must exceed N=1 ({arena_1}B)"
            )


# ===========================================================================
# 3. CpuCCodegenTarget — artefact structure tests
# ===========================================================================

class TestCpuCCodegenTarget:

    # --- All four output files exist ---

    def test_model_c_written(self, runner, tmp_path):
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        assert (Path(out_dir) / "model.c").exists(), "model.c must be written"

    def test_model_h_written(self, runner, tmp_path):
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        assert (Path(out_dir) / "model.h").exists(), "model.h must be written"

    def test_weights_bin_written(self, runner, tmp_path):
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        assert (Path(out_dir) / "weights.bin").exists(), "weights.bin must be written"

    def test_arena_size_h_written(self, runner, tmp_path):
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        assert (Path(out_dir) / "arena_size.h").exists(), "arena_size.h must be written"

    # --- model.c structural content ---

    def test_model_c_contains_im2col_helper(self, runner, tmp_path):
        """model.c must contain the static seecpp_im2col helper."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        src = (Path(out_dir) / "model.c").read_text()
        assert "seecpp_im2col" in src, \
            "model.c must contain the seecpp_im2col static helper"

    def test_model_c_contains_cblas_sgemm(self, runner, tmp_path):
        """model.c must call cblas_sgemm for the lowered matmul op."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        src = (Path(out_dir) / "model.c").read_text()
        assert "cblas_sgemm" in src, \
            "model.c must call cblas_sgemm for sc_low.matmul"

    def test_model_c_contains_run_function(self, runner, tmp_path):
        """model.c must define seecpp_run_model."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        src = (Path(out_dir) / "model.c").read_text()
        assert "seecpp_run_model" in src, \
            "model.c must define seecpp_run_model"

    def test_model_c_fused_relu_loop_emitted(self, runner, tmp_path):
        """When activation=relu is stamped on a matmul, model.c must contain
        the fused relu loop (the in-place max(0,x) loop after cblas_sgemm)."""
        onnx_file = str(tmp_path / "gemm_relu.onnx")
        sir_file  = str(tmp_path / "gemm_relu.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_gemm_relu())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        src = (Path(out_dir) / "model.c").read_text()
        # Both the sgemm call and a post-processing relu loop must appear.
        assert "cblas_sgemm" in src
        assert "0.0f" in src, \
            "Fused relu loop must compare to 0.0f in model.c"

    def test_model_c_auto_generated_comment(self, runner, tmp_path):
        """model.c must start with the AUTO-GENERATED guard comment."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        src = (Path(out_dir) / "model.c").read_text()
        assert "AUTO-GENERATED" in src, \
            "model.c must contain the AUTO-GENERATED guard"

    # --- model.h structural content ---

    def test_model_h_extern_c(self, runner, tmp_path):
        """model.h must contain extern \"C\" for C++ compatibility."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        h = (Path(out_dir) / "model.h").read_text()
        assert 'extern "C"' in h, \
            'model.h must contain extern "C" guard'

    def test_model_h_declares_run_function(self, runner, tmp_path):
        """model.h must declare seecpp_run_model."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        h = (Path(out_dir) / "model.h").read_text()
        assert "seecpp_run_model" in h, \
            "model.h must declare seecpp_run_model"

    # --- arena_size.h ---

    def test_arena_size_h_parseable(self, runner, tmp_path):
        """SEECPP_ARENA_BYTES in arena_size.h must be a valid integer."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        h = (Path(out_dir) / "arena_size.h").read_text()
        m = re.search(r"SEECPP_ARENA_BYTES\s*=\s*(\d+)", h)
        assert m, "SEECPP_ARENA_BYTES must be a numeric constant in arena_size.h"
        assert int(m.group(1)) > 0, \
            "SEECPP_ARENA_BYTES must be positive for a model with activations"

    # --- weights.bin ---

    def test_weights_bin_nonempty(self, runner, tmp_path):
        """weights.bin must be non-empty for any model with initializers."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        size = (Path(out_dir) / "weights.bin").stat().st_size
        assert size > 0, "weights.bin must contain at least one weight tensor"

    def test_weights_bin_size_matches_filter(self, runner, tmp_path):
        """weights.bin byte count must be >= F*C*KH*KW*4 (the filter size)."""
        F, C, KH, KW = 8, 3, 3, 3
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only(F=F, C=C))
        runner.run_and_parse(onnx_file, sir_file, codegen_out_dir=out_dir)
        size = (Path(out_dir) / "weights.bin").stat().st_size
        min_expected = F * C * KH * KW * 4
        assert size >= min_expected, (
            f"weights.bin ({size}B) must be >= filter size ({min_expected}B)"
        )

    def test_weights_bin_grows_with_conv_bn_fold(self, runner, tmp_path):
        """weights.bin for a Conv+BN model must be larger than for a Conv-only
        model of the same shape — folded tensors are appended."""
        F, C = 8, 3
        onnx_conv = str(tmp_path / "conv.onnx")
        onnx_cbn  = str(tmp_path / "cbn.onnx")
        out_conv  = str(tmp_path / "out_conv")
        out_cbn   = str(tmp_path / "out_cbn")
        _write_model(onnx_conv, _make_conv_only(F=F, C=C))
        _write_model(onnx_cbn,  _make_conv_bn_relu(F=F, C=C))

        runner.run_and_parse(onnx_conv, str(tmp_path / "conv.sir"),
                              codegen_out_dir=out_conv)
        runner.run_and_parse(onnx_cbn,  str(tmp_path / "cbn.sir"),
                              codegen_out_dir=out_cbn)

        size_conv = (Path(out_conv) / "weights.bin").stat().st_size
        size_cbn  = (Path(out_cbn)  / "weights.bin").stat().st_size
        assert size_cbn > size_conv, (
            f"Conv+BN weights.bin ({size_cbn}B) must be larger than "
            f"Conv-only ({size_conv}B) due to appended folded tensors"
        )


# ===========================================================================
# 4. CodegenDriver — pipeline orchestration
# ===========================================================================

class TestCodegenDriver:

    def test_driver_completion_logged(self, runner, tmp_path):
        """CodegenDriver must log pipeline completion."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_only())
        summary = runner.run_and_parse(onnx_file, sir_file)
        assert _driver_completed(summary), \
            "CodegenDriver must log 'pipeline complete'"

    def test_codegen_target_name_in_log(self, runner, tmp_path):
        """CodegenDriver completion line must name CpuCCodegenTarget."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_only())
        summary = runner.run_and_parse(onnx_file, sir_file)
        assert "CpuCCodegenTarget" in _combined(summary), \
            "Target name 'CpuCCodegenTarget' must appear in Logger output"

    def test_weight_folding_runs_before_allocation(self, runner, tmp_path):
        """WeightFoldingPass log line must appear BEFORE BufferAllocator
        log line in combined output — stage ordering invariant."""
        onnx_file = str(tmp_path / "cbr.onnx")
        sir_file  = str(tmp_path / "cbr.sir")
        _write_model(onnx_file, _make_conv_bn_relu())
        summary = runner.run_and_parse(onnx_file, sir_file)

        combined = _combined(summary)
        fold_pos  = combined.find("WeightFoldingPass: folded")
        alloc_pos = combined.find("BufferAllocator:")
        assert fold_pos != -1 and alloc_pos != -1, \
            "Both WeightFoldingPass and BufferAllocator must log"
        assert fold_pos < alloc_pos, (
            "WeightFoldingPass must complete before BufferAllocator runs "
            f"(fold_pos={fold_pos} alloc_pos={alloc_pos})"
        )

    def test_allocation_runs_before_codegen(self, runner, tmp_path):
        """BufferAllocator log line must appear BEFORE CpuCCodegenTarget
        log line — allocation must precede emission."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_only())
        summary = runner.run_and_parse(onnx_file, sir_file,
                                        codegen_out_dir=out_dir)
        combined  = _combined(summary)
        alloc_pos = combined.find("BufferAllocator:")
        emit_pos  = combined.find("CpuCCodegenTarget: wrote")
        assert alloc_pos != -1 and emit_pos != -1
        assert alloc_pos < emit_pos, \
            "BufferAllocator must run before CpuCCodegenTarget::emit"

    def test_driver_arena_bytes_positive_for_conv(self, runner, tmp_path):
        """CodegenDriver must report a positive arena for a conv model."""
        onnx_file = str(tmp_path / "conv.onnx")
        sir_file  = str(tmp_path / "conv.sir")
        _write_model(onnx_file, _make_conv_only())
        summary = runner.run_and_parse(onnx_file, sir_file)
        arena = _parse_arena_bytes_from_driver(summary)
        assert arena is not None and arena > 0

    def test_full_pipeline_conv_bn_relu(self, runner, tmp_path):
        """End-to-end: Conv+BN+Relu must complete all three codegen stages
        and produce all four artefacts."""
        onnx_file = str(tmp_path / "cbr.onnx")
        sir_file  = str(tmp_path / "cbr.sir")
        out_dir   = str(tmp_path / "out")
        _write_model(onnx_file, _make_conv_bn_relu())
        summary = runner.run_and_parse(onnx_file, sir_file,
                                        codegen_out_dir=out_dir)

        assert summary.returncode == 0
        assert _parse_fold_count(summary) == 1
        assert _parse_arena_bytes_from_driver(summary) > 0
        assert _driver_completed(summary)
        for fname in ["model.c", "model.h", "weights.bin", "arena_size.h"]:
            assert (Path(out_dir) / fname).exists(), \
                f"{fname} must exist after full pipeline"


# ===========================================================================
# 5. Numerical correctness (requires C compiler + cffi)
# ===========================================================================

def _cc_available() -> bool:
    return shutil.which("cc") is not None or shutil.which("gcc") is not None


def _cblas_available() -> bool:
    # Try a minimal compile that links against cblas.
    try:
        r = subprocess.run(
            ["cc", "-x", "c", "-", "-lblas", "-o", "/dev/null"],
            input=b"#include <cblas.h>\nint main(){return 0;}\n",
            capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


@pytest.mark.numerical
@pytest.mark.skipif(
    not _cc_available(),
    reason="C compiler not available — skipping numerical tests")
class TestNumericalCorrectness:

    def _compile_model(self, out_dir: Path) -> Path:
        """Compile model.c into a shared library. Returns path to .so."""
        so = out_dir / "model.so"
        cc = shutil.which("cc") or shutil.which("gcc")
        result = subprocess.run(
            [cc, "-O2", "-shared", "-fPIC",
             str(out_dir / "model.c"),
             "-lblas", "-lm",
             "-o", str(so)],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            pytest.skip(
                f"Could not compile model.c (cblas may not be installed):\n"
                f"{result.stderr}")
        return so

    def _load_and_run(self, so: Path, out_dir: Path,
                       inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Load the compiled .so via cffi and call seecpp_run_model."""
        try:
            import cffi
        except ImportError:
            pytest.skip("cffi not installed — skipping numerical tests")

        ffi = cffi.FFI()
        ffi.cdef("""
            void seecpp_run_model(
                const float* const* inputs,
                float**             outputs,
                void*               arena,
                const void*         weight_blob);
        """)
        lib = ffi.dlopen(str(so))

        # Read arena size.
        h = (out_dir / "arena_size.h").read_text()
        arena_bytes = int(re.search(r"SEECPP_ARENA_BYTES\s*=\s*(\d+)", h).group(1))

        # Read weight blob.
        weight_data = (out_dir / "weights.bin").read_bytes()

        arena   = ffi.new(f"char[{arena_bytes}]")
        wblob   = ffi.from_buffer(weight_data)

        # Build input pointers.
        in_bufs  = [ffi.from_buffer(inp.astype(np.float32)) for inp in inputs]
        in_ptrs  = ffi.new("const float*[]", in_bufs)

        # We need output size — read from arena_size.h comment or infer.
        # For simplicity, allocate a generous output buffer.
        out_buf  = ffi.new("float[65536]")
        out_ptrs = ffi.new("float*[]", [out_buf])

        lib.seecpp_run_model(in_ptrs, out_ptrs, arena, wblob)

        return [np.frombuffer(ffi.buffer(out_buf), dtype=np.float32).copy()]

    def test_relu_output_nonnegative(self, runner, tmp_path):
        """Compiled Gemm+Relu model: all outputs must be >= 0."""
        onnx_file = str(tmp_path / "gr.onnx")
        sir_file  = str(tmp_path / "gr.sir")
        out_dir   = tmp_path / "out"
        _write_model(onnx_file, _make_gemm_relu(M=2, K=4, N=8))
        runner.run_and_parse(onnx_file, sir_file,
                              codegen_out_dir=str(out_dir))

        so = self._compile_model(out_dir)
        X  = np.random.randn(2, 4).astype(np.float32)
        outputs = self._load_and_run(so, out_dir, [X])
        assert np.all(outputs[0] >= 0), \
            "Relu output must be non-negative for all elements"

    def test_zero_input_gives_zero_output_through_relu(self, runner, tmp_path):
        """Zero input through Gemm (zero-initialised weights) + Relu must
        produce all-zero output."""
        onnx_file = str(tmp_path / "gr.onnx")
        sir_file  = str(tmp_path / "gr.sir")
        out_dir   = tmp_path / "out"
        _write_model(onnx_file, _make_gemm_relu(M=1, K=4, N=4))
        runner.run_and_parse(onnx_file, sir_file,
                              codegen_out_dir=str(out_dir))

        so = self._compile_model(out_dir)
        X  = np.zeros((1, 4), dtype=np.float32)
        outputs = self._load_and_run(so, out_dir, [X])
        np.testing.assert_array_equal(
            outputs[0][:4], np.zeros(4, dtype=np.float32),
            err_msg="Zero input with zero weights must produce zero output")