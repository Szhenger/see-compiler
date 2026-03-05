#include "include/backend/codegen_driver.hpp"
#include "include/backend/cpu_c_codegen_target.hpp"
#include "include/utility/logger.hpp"

// =============================================================================
// Backend Pipeline Assembly
//
// All #includes of concrete backend types live here — not in the header.
// Callers that use buildCpuCCodegen() only pay for recompilation when
// the CpuCCodegenTarget implementation changes, not the CodegenDriver
// or WeightFoldingPass interfaces.
//
// Pipeline:
//   Frontend + Middle-end output (sc_low.*, fully shaped)
//        │
//        ▼
//   [1]  WeightFoldingPass
//        Reads fused_bn attributes stamped by OperatorFusionPass.
//        Computes W_fused = W * scale / sqrt(var + eps) per output channel.
//        Writes folded tensors to WeightBuffer under *__bn_folded keys.
//        Updates Conv op attributes so codegen reads folded tensors.
//        │
//        ▼
//   [2]  BufferAllocator
//        Linear-scan allocation over the sc_low.* Value sequence.
//        Assigns arena offsets to activations; marks weight Values is_weight.
//        Produces BufferAllocation map + total arena byte count.
//        │
//        ▼
//   [3]  CpuCCodegenTarget::emit
//        Emits model.c (cblas_sgemm + hand-written im2col),
//               model.h (public seecpp_run_model() declaration),
//               weights.bin (flat weight blob),
//               arena_size.h (SEECPP_ARENA_BYTES constant).
//        │
//        ▼
//   Output artefacts ready for downstream compilation:
//     cc -O3 -march=native model.c -lblas -shared -fPIC -o model.so
//
// =============================================================================

namespace seecpp::backend {

[[nodiscard]]
CodegenDriver buildCpuCCodegen(const std::filesystem::path& output_dir) {
    utility::Logger::info(
        "Backend pipeline: CpuCCodegenTarget -> '" +
        output_dir.string() + "'");

    return CodegenDriver(
        std::make_unique<CpuCCodegenTarget>(output_dir));
}

} // namespace seecpp::backend