#pragma once

#include "middle-end/pass_manager.hpp"
#include <cstdint>

namespace seecpp::middle {

enum class OptLevel : uint8_t {
    O0, 
    O1, 
    O2, 
    O3 
};

struct PipelineConfig {
    OptLevel level{OptLevel::O2};
    size_t l1_cache_size{32768};
    bool enable_autodiff{true};
    bool strict_continuity_check{true};
};

/**
 * Pipeline Order:
 * 1. HighLevelFusion (Conv+BN)
 * 2. DCE
 * 3. ConvLowering (Conv -> Im2Col + MatMul)
 * 4. LowLevelFusion (MatMul+ReLU)
 * 5. DCE
 */

[[nodiscard]] PassManager buildMiddleEndPipeline(const PipelineConfig& config = {});

} // namespace seecpp::middle
