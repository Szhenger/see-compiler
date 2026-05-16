#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace seecpp::backend {

/// Defines the target hardware architecture for explicit SIMD generation.
enum class TargetArch {
    HostNative,     ///< Auto-detect from the compiling host
    x86_64_AVX2,
    x86_64_AVX512,
    ARM_NEON
};

/// Exposes all compiler heuristics and hardware-specific tuning parameters.
struct CodegenOptions {
    TargetArch arch = TargetArch::HostNative;
    
    // Hardware constraints for loop-tiling heuristics
    size_t l1_cache_size_kb = 32;
    size_t l2_cache_size_kb = 512;
    
    // Execution constraints
    bool emit_multithreaded = true;  // Emits std::execution::par policies
    bool enable_fast_math = false;   // Allows FMA contraction at the cost of strict IEEE-754

    // Debugging
    bool emit_debug_symbols = false;
};

[[nodiscard]]
std::unique_ptr<CodegenDriver> createCpuDriver(
    const std::filesystem::path& output_dir,
    const CodegenOptions& options = CodegenOptions{}
);

} // namespace seecpp::backend
