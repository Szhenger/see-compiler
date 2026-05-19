// source/backend/targets/cpu_avx512_target.cpp
#include "include/backend/targets/cpu_avx512_target.hpp"
#include <format>

namespace seecpp::backend {

CpuAvx512Target::CpuAvx512Target(std::filesystem::path output_dir, CodegenOptions options)
    : CpuGenericCTarget(std::move(output_dir), std::move(options)) {}

void CpuAvx512Target::emit(const ir::SIRGraph& graph, const BufferAllocationMap& arena) {
    // AVX-512 requires <immintrin.h> in the emitted source
    // We would prepend this to the source file string builder here
    CpuGenericCTarget::emit(graph, arena); 
}

std::string CpuAvx512Target::emitElementwise(const ir::Node& node) {
    size_t size = node.size();
    size_t vec_loops = size / kSimdWidth;
    size_t remainder = size % kSimdWidth;

    std::string code = std::format(R"(
    // AVX-512 Node: {} (Size: {}, Unrolled: {})
    float* out = reinterpret_cast<float*>(&arena->buffer[{}]);
    const float* in1 = reinterpret_cast<const float*>(&arena->buffer[{}]);
    const float* in2 = reinterpret_cast<const float*>(&arena->buffer[{}]);

    size_t i = 0;
    for (; i < {}; i += 16) {{
        __m512 v1 = _mm512_load_ps(&in1[i]);
        __m512 v2 = _mm512_load_ps(&in2[i]);
        __m512 vr = _mm512_add_ps(v1, v2); // Example: Vector Add
        _mm512_store_ps(&out[i], vr);
    }}
)", node.name(), size, vec_loops, node.out_offset(), node.in1_offset(), node.in2_offset(), vec_loops * kSimdWidth);

    // If the tensor size isn't a perfect multiple of 16, we emit AVX-512 Masked intrinsics
    if (remainder > 0) {
        code += std::format(R"(
    // Tail Mask Processing (Remainder: {})
    __mmask16 tail_mask = (1U << {}) - 1;
    __m512 v1_tail = _mm512_maskz_load_ps(tail_mask, &in1[i]);
    __m512 v2_tail = _mm512_maskz_load_ps(tail_mask, &in2[i]);
    __m512 vr_tail = _mm512_add_ps(v1_tail, v2_tail);
    _mm512_mask_store_ps(&out[i], tail_mask, vr_tail);
)", remainder, remainder);
    }

    return code;
}

} // namespace seecpp::backend
