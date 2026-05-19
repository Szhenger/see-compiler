// source/backend/targets/cpu_neon_target.cpp
#include "include/backend/targets/cpu_neon_target.hpp"
#include <format>

namespace seecpp::backend {

CpuNeonTarget::CpuNeonTarget(std::filesystem::path output_dir, CodegenOptions options)
    : CpuGenericCTarget(std::move(output_dir), std::move(options)) {}

void CpuNeonTarget::emit(const ir::SIRGraph& graph, const BufferAllocationMap& arena) {
    // ARM NEON requires <arm_neon.h>
    CpuGenericCTarget::emit(graph, arena); 
}

std::string CpuNeonTarget::emitElementwise(const ir::Node& node) {
    size_t size = node.size();
    size_t vec_loops = size / kSimdWidth;
    size_t remainder = size % kSimdWidth;

    std::string code = std::format(R"(
    // NEON Node: {} (Size: {}, Unrolled: {})
    float* out = reinterpret_cast<float*>(&arena->buffer[{}]);
    const float* in1 = reinterpret_cast<const float*>(&arena->buffer[{}]);
    const float* in2 = reinterpret_cast<const float*>(&arena->buffer[{}]);

    size_t i = 0;
    for (; i < {}; i += 4) {{
        float32x4_t v1 = vld1q_f32(&in1[i]);
        float32x4_t v2 = vld1q_f32(&in2[i]);
        float32x4_t vr = vaddq_f32(v1, v2); // Example: Vector Add
        vst1q_f32(&out[i], vr);
    }}
)", node.name(), size, vec_loops, node.out_offset(), node.in1_offset(), node.in2_offset(), vec_loops * kSimdWidth);

    // Standard NEON lacks masked loads, so we fall back to a scalar loop for the tail
    if (remainder > 0) {
        code += std::format(R"(
    // Scalar Tail Processing (Remainder: {})
    for (; i < {}; ++i) {{
        out[i] = in1[i] + in2[i];
    }}
)", remainder, size);
    }

    return code;
}

} // namespace seecpp::backend
