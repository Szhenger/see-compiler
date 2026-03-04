#pragma once

#include "middle-end/pass_manager.hpp"

// =============================================================================
// Middle-End Pipeline Assembly
//
// Pass ordering:
//   [1] OperatorFusionPass    — Conv+BN fusion (sc_high, pre-lowering)
//   [2] DeadCodeElimination   — remove BN ops absorbed by fusion
//   [3] ConvLoweringPass      — sc_high.conv2d -> sc_low.im2col + sc_low.matmul
//   [4] OperatorFusionPass    — MatMul+Relu fusion (sc_low, post-lowering)
//   [5] DeadCodeElimination   — remove residual dead ops
// =============================================================================

namespace seecpp::middle {

/// Build and return the canonical SeeC++ middle-end PassManager.
/// Caller drives execution with pm.run(block).
[[nodiscard]]
PassManager buildMiddleEndPipeline();

} // namespace seecpp::middle