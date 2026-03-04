#include "include/middle-end/middle_end_driver.hpp"
#include "include/middle-end/conv_lowering_pass.hpp"
#include "include/middle-end/op_fusion_pass.hpp"
#include "include/middle-end/dead_code_elimin.hpp"
#include "include/utility/logger.hpp"

// =============================================================================
// Middle-End Pipeline Assembly
//
// All #includes of concrete pass types live here — not in the header.
// The header exposes only PassManager (an abstract pipeline handle) so
// translation units that call buildMiddleEndPipeline() do not need to
// recompile when a pass implementation changes.
//
// Pass ordering rationale:
//
//   Frontend output (sc_high.*, fully shaped and validated)
//        │
//        ▼
//   [1]  OperatorFusionPass      (sc_high, pre-lowering)
//        Conv+BN fusion MUST run before ConvLoweringPass.
//        Once Conv is decomposed into im2col+matmul the BN has
//        no adjacent Conv to fold into.
//        │
//        ▼
//   [2]  DeadCodeEliminationPass
//        BN ops absorbed by fusion are now dead. Pruning them here
//        keeps ConvLoweringPass from encountering stale BN-linked ops.
//        │
//        ▼
//   [3]  ConvLoweringPass
//        sc_high.conv2d  →  sc_low.im2col + sc_low.reshape + sc_low.matmul
//        All convolutions in the block are lowered in a single sweep.
//        │
//        ▼
//   [4]  OperatorFusionPass      (sc_low, post-lowering)
//        A second fusion sweep catches sc_low.matmul + relu patterns
//        that only become visible after Conv lowering.
//        │
//        ▼
//   [5]  DeadCodeEliminationPass
//        Final prune of any ops orphaned by the second fusion sweep.
//        │
//        ▼
//   Middle-end output (sc_low.*, ready for backend code generation)
//
// =============================================================================

namespace seecpp::middle {

[[nodiscard]]
PassManager buildMiddleEndPipeline() {
    PassManager pm;

    // [1] High-level fusion: Conv+BN folding, elementwise chain merging.
    pm.addPass<OperatorFusionPass>();

    // [2] Remove BN ops made dead by fusion before lowering touches them.
    pm.addPass<DeadCodeEliminationPass>();

    // [3] Lower all sc_high.conv2d ops to sc_low GEMM decomposition.
    pm.addPass<ConvLoweringPass>();

    // [4] Post-lowering fusion: absorb relu activations into matmul ops.
    pm.addPass<OperatorFusionPass>();

    // [5] Final dead-code sweep.
    pm.addPass<DeadCodeEliminationPass>();

    utility::Logger::info(
        "Middle-end pipeline built: " +
        std::to_string(pm.numPasses()) + " passes registered");

    return pm;
}

} // namespace seecpp::middle