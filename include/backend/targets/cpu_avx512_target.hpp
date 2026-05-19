// include/backend/targets/cpu_avx512_target.hpp
#pragma once

#include "backend/targets/cpu_generic_c_target.hpp"
#include <filesystem>
#include <string>

namespace seecpp::backend {

/// Emits explicit Intel AVX-512 intrinsic code.
/// Targets high-throughput datacenter training.
class CpuAvx512Target : public CpuGenericCTarget {
public:
    explicit CpuAvx512Target(std::filesystem::path output_dir, CodegenOptions options);
    ~CpuAvx512Target() override = default;

    void emit(const ir::SIRGraph& graph, const BufferAllocationMap& arena) override;

protected:
    // Hardware constraints: AVX-512 requires 64-byte aligned memory access
    // The BufferAllocator MUST respect this, or _mm512_load_ps will segfault.
    static constexpr size_t kAlignmentBytes = 64;
    static constexpr size_t kSimdWidth = 16; // 16 floats per vector

    // Micro-Kernel Overrides
    // Emits register-blocked loops using _mm512_fmadd_ps
    [[nodiscard]] std::string emitGEMM(const ir::Node& node) override;
    
    // Emits masked loops for remainder dimensions (Tail processing)
    [[nodiscard]] std::string emitElementwise(const ir::Node& node) override;

private:
    // AVX-512 specific utility to generate tail-mask logic
    [[nodiscard]] std::string generateMaskLogic(size_t remainder_size);
};

} // namespace seecpp::backend
