#pragma once

#include "backend/codegen.hpp"

// =============================================================================
// Backend Pipeline Assembly
//
// Symmetric to middle_end_driver.hpp — declaration only here,
// all concrete type includes live in backend_driver.cpp.
//
// Backend pipeline:
//   [1] WeightFoldingPass    — fold BN params into Conv weights
//   [2] BufferAllocator      — assign arena offsets to all Values
//   [3] ICodegenTarget::emit — emit C / CUDA / other target code
// =============================================================================

namespace seecpp::backend {

/// Build a CodegenDriver targeting CPU C codegen.
/// Output artefacts are written to `output_dir`.
[[nodiscard]]
CodegenDriver buildCpuCCodegen(const std::filesystem::path& output_dir);

} // namespace seecpp::backend