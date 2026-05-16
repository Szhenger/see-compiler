#include "include/backend/codegen_driver.hpp"

// Pass definitions
#include "include/backend/passes/weight_folding_pass.hpp"
#include "include/backend/passes/buffer_allocator.hpp"

// Concrete target emitters
#include "include/backend/targets/cpu_generic_c_target.hpp"
#include "include/backend/targets/cpu_avx512_target.hpp"
#include "include/backend/targets/cpu_neon_target.hpp"

// Utilities
#include "include/utility/logger.hpp"

#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>

namespace seecpp::backend {

// =============================================================================
// Backend Pipeline Assembly
//
// Pipeline:
//   Frontend + Middle-end output (sc_low.*, fully shaped)
//        │
//        ▼
//   [1]  WeightFoldingPass
//   [2]  BufferAllocator
//   [3]  ICodegenTarget::emit (Polymorphic based on TargetArch)
// =============================================================================

[[nodiscard]]
std::unique_ptr<CodegenDriver> createCpuDriver(
    const std::filesystem::path& output_dir,
    const CodegenOptions& options) 
{
    // Step 1: Ensure Output Directory Exists
    std::error_code ec;
    if (!std::filesystem::exists(output_dir) && !std::filesystem::create_directories(output_dir, ec)) {
        throw std::invalid_argument(
            std::format("Failed to create output directory '{}': {}", output_dir.string(), ec.message())
        );
    }

    utility::Logger::info(
        std::format("Initializing Backend Pipeline. Target Directory: {}", output_dir.string())
    );

    // Step 2: Resolve Target Architecture & Instantiate Specific Emitter
    std::unique_ptr<ICodegenTarget> target_emitter;

    switch (options.arch) {
        case TargetArch::HostNative:
            // TODO: Query CPUID here. Defaulting to generic C for safety.
            utility::Logger::warn("HostNative target selected. Defaulting to generic C code.");
            target_emitter = std::make_unique<CpuGenericCTarget>(output_dir, options);
            break;

        case TargetArch::x86_64_AVX512:
            utility::Logger::info("Targeting x86_64 AVX-512 Intrinsics.");
            target_emitter = std::make_unique<CpuAvx512Target>(output_dir, options);
            break;

        case TargetArch::ARM_NEON:
            utility::Logger::info("Targeting ARM NEON Intrinsics.");
            target_emitter = std::make_unique<CpuNeonTarget>(output_dir, options);
            break;

        case TargetArch::x86_64_AVX2:
        default:
            throw std::invalid_argument("Requested TargetArch is currently unsupported by the backend.");
    }

    // Step 3: Assemble the Driver Pipeline
    // A robust compiler driver explicitly registers its passes in order of execution.
    auto driver = std::make_unique<CodegenDriver>(std::move(target_emitter));

    driver->add_pass(std::make_unique<WeightFoldingPass>());
    driver->add_pass(std::make_unique<BufferAllocator>(
        options.l1_cache_size_kb, 
        options.l2_cache_size_kb
    ));

    return driver;
}

} // namespace seecpp::backend
