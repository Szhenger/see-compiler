#pragma once

#include "backend/codegen.hpp" // Pulls in ICodegenTarget, CodegenOptions, AllocationResult, BackendError
#include "middle-end/SIR.hpp"
#include "utility/weight_buffer.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <ostream>
#include <expected>

namespace seecpp::backend {

class CpuCCodegenTarget : public ICodegenTarget {
public:
    explicit CpuCCodegenTarget(std::filesystem::path output_dir);
    ~CpuCCodegenTarget() override = default;

    [[nodiscard]] std::string_view name() const override { return "CpuCCodegenTarget"; }

    /// Primary entry point orchestrating disk compilation passes.
    [[nodiscard]] std::expected<void, BackendError> emit(
        const sir::Block& block,
        const AllocationResult& allocation,
        const utility::WeightBuffer& weights,
        const CodegenOptions& options) override;

protected:
    // Core structural generators
    [[nodiscard]] virtual std::expected<void, BackendError> emitSourceFile(
        const sir::Block& block,
        const AllocationResult& allocation,
        const utility::WeightBuffer& weights,
        const CodegenOptions& options);

    [[nodiscard]] virtual std::expected<void, BackendError> emitHeaderFile(
        const sir::Block& block,
        const AllocationResult& allocation,
        const CodegenOptions& options);

    // --- Targeted Operation Emitters ---
    // These return explicit string segments allowing derived SIMD classes 
    // to cherry-pick and override specialized micro-kernels.

    [[nodiscard]] virtual std::string emitIm2ColHelper(const CodegenOptions& options);
    
    [[nodiscard]] virtual std::string emitIm2ColOp(const sir::Operation* op, const AllocationResult& alloc);
    [[nodiscard]] virtual std::string emitMatMulOp(const sir::Operation* op, const AllocationResult& alloc, const CodegenOptions& options);
    [[nodiscard]] virtual std::string emitAddOp(const sir::Operation* op, const AllocationResult& alloc);
    [[nodiscard]] virtual std::string emitReluOp(const sir::Operation* op, const AllocationResult& alloc);
    [[nodiscard]] virtual std::string emitFusedEwOp(const sir::Operation* op, const AllocationResult& alloc);

private:
    std::filesystem::path output_dir_;

    // --- Internal Code Generation Utilities ---
    
    [[nodiscard]] std::expected<void, BackendError> writeWeightsBin(const utility::WeightBuffer& weights);

    /// Formats a safe C-compliant naming identifier mapping an SSA value string.
    [[nodiscard]] static std::string sanitizeCName(const sir::Value* v);

    /// Resolves data offset addressing rules directly translating to physical pointer expressions.
    [[nodiscard]] static std::string getPointerExpression(
        const sir::Value* v,
        const AllocationResult& alloc,
        std::string_view arena_var,
        std::string_view weights_var);
};

} // namespace seecpp::backend
