#pragma once

#include "backend/codegen.hpp"

#include <filesystem>
#include <string>

// =============================================================================
// CpuCCodegenTarget
//
// Emits a self-contained C99 source file that implements the sc_low.*
// op sequence using:
//   - A hand-written im2col loop (no external dependency)
//   - cblas_sgemm for sc_low.matmul (OpenBLAS / Accelerate / MKL)
//   - In-place loops for sc_low.add, sc_low.relu, sc_high.fused_ew
//
// Output artefacts (written to output_dir_):
//   model.c       — the generated inference function
//   model.h       — public header declaring seecpp_run_model()
//   weights.bin   — flat binary dump of all WeightBuffer tensors
//   arena_size.h  — constexpr size_t SEECPP_ARENA_BYTES = N;
//
// The generated model.c implements one function:
//
//   void seecpp_run_model(
//       const float* __restrict__ inputs[],   // one ptr per graph input
//       float*       __restrict__ outputs[],  // one ptr per graph output
//       void*        arena,                   // scratch memory (arena_bytes)
//       const void*  weights                  // flat weight blob
//   );
//
// Compilation (example):
//   cc -O3 -march=native model.c -lblas -o model.so -shared -fPIC
//
// Design notes:
//   - Every Value maps to a pointer expression into `arena` or `weights`.
//   - Im2Col is emitted as a static helper function at the top of model.c.
//   - BatchNorm-folded convolutions use the __bn_folded weight keys.
//   - The activation=relu attribute on sc_low.matmul is emitted as a
//     fused post-processing loop immediately after the cblas_sgemm call.
//   - sc_high.fused_ew ops emit a single loop over the op_sequence attribute.
// =============================================================================

namespace seecpp::backend {

class CpuCCodegenTarget final : public ICodegenTarget {
public:
    explicit CpuCCodegenTarget(std::filesystem::path output_dir)
        : output_dir_(std::move(output_dir)) {}

    std::string_view name() const override { return "CpuCCodegenTarget"; }

    [[nodiscard]]
    std::expected<void, CodegenError> emit(
        const sir::Block&            block,
        const BufferAllocation&      allocation,
        const utility::WeightBuffer& weights) override;

private:
    std::filesystem::path output_dir_;

    // --- Emission helpers ---

    /// Emit the static im2col helper at the top of model.c.
    void emitIm2ColHelper(std::ostream& os);

    /// Emit a pointer declaration mapping `v` to its arena/weight slot.
    void emitValuePtr(std::ostream&           os,
                      const sir::Value*       v,
                      const BufferAllocation& alloc,
                      const std::string&      arena_var,
                      const std::string&      weights_var);

    /// Emit code for a single sc_low.* or sc_high.fused_ew op.
    [[nodiscard]]
    std::expected<void, CodegenError>
    emitOp(std::ostream&                os,
           const sir::Operation*        op,
           const BufferAllocation&      alloc,
           const utility::WeightBuffer& weights);

    /// Emit sc_low.im2col as a call to the static helper.
    void emitIm2ColOp(std::ostream& os, const sir::Operation* op,
                      const BufferAllocation& alloc);

    /// Emit sc_low.matmul as cblas_sgemm (with optional fused relu).
    void emitMatMulOp(std::ostream& os, const sir::Operation* op,
                      const BufferAllocation& alloc);

    /// Emit sc_low.add as an element-wise loop.
    void emitAddOp(std::ostream& os, const sir::Operation* op,
                   const BufferAllocation& alloc);

    /// Emit sc_high.relu / sc_low.relu as a max(0,x) loop.
    void emitReluOp(std::ostream& os, const sir::Operation* op,
                    const BufferAllocation& alloc);

    /// Emit sc_high.fused_ew as a single fused loop.
    void emitFusedEwOp(std::ostream& os, const sir::Operation* op,
                       const BufferAllocation& alloc);

    /// Emit sc_high.gemm as cblas_sgemm.
    void emitGemmOp(std::ostream& os, const sir::Operation* op,
                    const BufferAllocation& alloc);

    /// Write the flat weight binary (weights.bin).
    [[nodiscard]]
    std::expected<void, CodegenError>
    writeWeightsBin(const utility::WeightBuffer& weights);

    /// Write arena_size.h.
    void writeArenaSizeHeader(size_t arena_bytes);

    /// Write model.h (public inference API declaration).
    void writeModelHeader(const sir::Block& block);

    // --- Naming helpers ---

    /// Return a C identifier for a Value (strips % prefix from SSA ids).
    static std::string cName(const sir::Value* v);

    /// Return the C pointer expression for a Value given its slot.
    static std::string ptrExpr(const sir::Value*       v,
                                const BufferAllocation& alloc,
                                const std::string&      arena_var,
                                const std::string&      weights_var);
};

} // namespace seecpp::backend