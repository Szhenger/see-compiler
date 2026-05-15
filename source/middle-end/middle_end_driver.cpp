#include "middle-end/middle_end_driver.hpp"
#include "middle-end/conv_lowering_pass.hpp"
#include "middle-end/operator_fusion_pass.hpp"
#include "middle-end/dead_code_elimination_pass.hpp"

// Forward declaration for conceptual Autodiff & Memory passes discussed earlier
// #include "middle-end/autodiff_pass.hpp"
// #include "middle-end/memory_arena_pass.hpp"

namespace seecpp::middle {

[[nodiscard]] PassManager buildMiddleEndPipeline(const PipelineConfig& config) {
    PassManager pm;
    if (config.level >= OptLevel::O2) {
        // Fuses Conv2D + BatchNorm before structural lowering destroys the context
        pm.addPass<OperatorFusionPass>(FusionScope::HighLevelDialect);
    }
    if (config.level >= OptLevel::O1) {
        // Prunes defunct BN weights and orphaned structures
        pm.addPass<DeadCodeEliminationPass>();
    }
    pm.addPass<ConvLoweringPass>();
    if (config.level >= OptLevel::O2) {
        // Fuses the newly lowered MatMul with adjacent ReLUs
        pm.addPass<OperatorFusionPass>(FusionScope::LowLevelDialect);
    }
    if (config.level >= OptLevel::O1) {
        // Final recursive prune of structural metadata offsets prior to memory mapping
        pm.addPass<DeadCodeEliminationPass>();
    }
    return pm;
}

} // namespace seecpp::middle
