// include/backend/targets/cpu_neon_target.hpp
#pragma once

#include "backend/targets/cpu_generic_c_target.hpp"
#include <filesystem>
#include <string>

namespace seecpp::backend {

/// Emits explicit ARM NEON intrinsic code.
/// Targets edge devices, Apple Silicon, and mobile processors.
class CpuNeonTarget : public CpuGenericCTarget {
public:
    explicit CpuNeonTarget(std::filesystem::path output_dir, CodegenOptions options);
    ~CpuNeonTarget() override = default;

    void emit(const ir::SIRGraph& graph, const BufferAllocationMap& arena) override;

protected:
    // Hardware constraints: NEON requires 16-byte aligned memory access
    static constexpr size_t kAlignmentBytes = 16;
    static constexpr size_t kSimdWidth = 4; // 4 floats per vector

    // Micro-Kernel Overrides
    // Emits loops using vmlaq_f32 (Vector Multiply Accumulate)
    [[nodiscard]] std::string emitGEMM(const ir::Node& node) override;
    [[nodiscard]] std::string emitElementwise(const ir::Node& node) override;
};

} // namespace seecpp::backend
