#pragma once

#include "middle-end/sir.hpp"
#include "utility/weight_buffer.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <expected>
#include <memory>
#include <vector>

// =============================================================================
// SeeC++ Codegen — Core Abstractions
// =============================================================================

namespace seecpp::backend {

// =============================================================================
// Global Configuration
// =============================================================================

enum class TargetArch {
    HostNative,
    x86_64_AVX2,
    x86_64_AVX512,
    ARM_NEON
};

struct CodegenOptions {
    TargetArch arch = TargetArch::HostNative;
    size_t l1_cache_size_kb = 32;
    size_t l2_cache_size_kb = 512;
    size_t required_alignment = 64; // Crucial for SIMD (64 bytes for AVX-512)
    bool emit_multithreaded = true;
    bool enable_fast_math = false;
};

// =============================================================================
// Error Handling
// =============================================================================

struct BackendError {
    std::string stage;    // e.g., "weight_folding", "buffer_alloc", "target_emit"
    std::string message;
};

// =============================================================================
// Memory Allocation
// =============================================================================

struct BufferSlot {
    size_t offset = 0;       // Byte offset from arena base
    size_t byte_size = 0;    // Number of bytes this value occupies
    size_t alignment = 4;    // The byte-boundary this slot starts on
    bool is_weight = false;  // True => content lives in WeightBuffer
};

using BufferAllocationMap = std::unordered_map<const sir::Value*, BufferSlot>;

struct AllocationResult {
    BufferAllocationMap allocation;
    size_t arena_bytes = 0;  // Total activation arena needed, padded for alignment
};

// =============================================================================
// Interfaces
// =============================================================================

/// Abstract interface for pre-codegen mutation passes (e.g., WeightFolding, Quantization).
class IBackendPass {
public:
    virtual ~IBackendPass() = default;
    
    [[nodiscard]] virtual std::string_view name() const = 0;
    
    [[nodiscard]] virtual std::expected<void, BackendError> run(
        sir::Block& block,
        utility::WeightBuffer& weights,
        const CodegenOptions& options) = 0;
};

/// Abstract base for all code emission targets.
class ICodegenTarget {
public:
    virtual ~ICodegenTarget() = default;

    [[nodiscard]] virtual std::string_view name() const = 0;

    [[nodiscard]] virtual std::expected<void, BackendError> emit(
        const sir::Block& block,
        const AllocationResult& allocation,
        const utility::WeightBuffer& weights,
        const CodegenOptions& options) = 0;
};

// =============================================================================
// CodegenDriver — Orchestrator
// =============================================================================

/// Executes the backend pipeline.
/// 
/// Pipeline Architecture:
///   1. Mutating Passes (WeightFolding, Quantization, etc.)
///   2. Buffer Allocation (Determines hardware-aligned Arena offsets)
///   3. Emission (ICodegenTarget translates to text/binary artifact)
class CodegenDriver {
public:
    CodegenDriver(std::unique_ptr<ICodegenTarget> target, CodegenOptions options);

    /// Appends a mutation pass to the pipeline.
    void add_pass(std::unique_ptr<IBackendPass> pass);

    /// Runs the complete pipeline.
    [[nodiscard]] std::expected<void, BackendError> run(
        sir::Block& block, 
        utility::WeightBuffer& weights);

    /// Total padded byte count of the arena from the most recent run().
    [[nodiscard]] size_t arenaBytes() const { return latest_allocation_.arena_bytes; }

private:
    std::unique_ptr<ICodegenTarget> target_;
    CodegenOptions options_;
    std::vector<std::unique_ptr<IBackendPass>> passes_;
    AllocationResult latest_allocation_;

    // Internal step: Maps values to offsets respecting options_.required_alignment
    [[nodiscard]] std::expected<AllocationResult, BackendError> allocateBuffers(
        const sir::Block& block,
        const utility::WeightBuffer& weights);
};

} // namespace seecpp::backend
