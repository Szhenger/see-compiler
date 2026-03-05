#pragma once

#include "middle-end/SIR.hpp"
#include "utility/weight_buffer.hpp"
#include "utility/logger.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <expected>

// =============================================================================
// SeeC++ Codegen — Core Abstractions
//
// Three responsibilities of the codegen layer:
//
//   1. WeightFoldingPass (pre-codegen)
//      Reads fused_bn attributes stamped by OperatorFusionPass and performs
//      the actual arithmetic: W_fused = W * scale / sqrt(var + eps).
//      Writes folded tensors back to WeightBuffer under new keys.
//      This is deferred to the backend because it requires WeightBuffer
//      access, which is not available to middle-end passes.
//
//   2. BufferAllocator
//      Assigns a flat arena offset to every sc_low.* Value.
//      Uses a linear-scan algorithm over the forward op sequence.
//      Produces a BufferAllocation map: Value* -> {offset, byte_size}.
//
//   3. ICodegenTarget (interface)
//      Abstract base for all code emission targets.
//      Current implementation: CpuCCodegenTarget (C + cblas_sgemm).
//      Future: CudaCodegenTarget, VulkanCodegenTarget.
//
// Namespace: seecpp::backend
// =============================================================================

namespace seecpp::backend {

// =============================================================================
// Error type
// =============================================================================

struct BackendError {
    std::string stage;    // "weight_folding" | "buffer_alloc" | "codegen"
    std::string message;
};

// =============================================================================
// BufferAllocation — flat arena slot assigned to one Value
// =============================================================================

struct BufferSlot {
    size_t  offset    = 0;   // byte offset from arena base
    size_t  byte_size = 0;   // number of bytes this value occupies
    bool    is_weight = false; // true => content lives in WeightBuffer, not arena
};

/// Maps every live Value in the block to a physical buffer slot.
/// Weights (sc_high.constant results) are annotated is_weight=true
/// and point into WeightBuffer rather than the activation arena.
using BufferAllocation = std::unordered_map<const sir::Value*, BufferSlot>;

// =============================================================================
// BufferAllocator
// =============================================================================

/// Linear-scan allocator over the sc_low.* op sequence.
///
/// Algorithm:
///   1. Walk ops in forward order.
///   2. On first use of a Value, assign it an arena slot at current watermark.
///   3. Advance watermark by the value's byte_size.
///   4. sc_high.constant results are flagged is_weight; they don't consume
///      arena space (they live in WeightBuffer's contiguous blob).
///
/// Output: BufferAllocation + total arena size in bytes.
class BufferAllocator {
public:
    struct Result {
        BufferAllocation allocation;
        size_t           arena_bytes = 0;  // total activation arena needed
    };

    /// Run the allocator over `block`.
    /// `weights` is consulted to identify which Values are weight tensors.
    [[nodiscard]]
    std::expected<Result, BackendError>
    allocate(const sir::Block& block,
             const utility::WeightBuffer& weights);

private:
    /// Assign a slot for `v` if not already assigned. Returns the slot.
    BufferSlot& assignSlot(const sir::Value*    v,
                            BufferAllocation&    alloc,
                            size_t&              watermark,
                            const utility::WeightBuffer& weights);
};

// =============================================================================
// ICodegenTarget — abstract code emission interface
// =============================================================================

/// Every code generation backend implements this interface.
///
/// Lifecycle:
///   1. Construct with output path.
///   2. Call emit(block, allocation, weights) — produces the output artifact.
///   3. Read diagnostics from Logger.
///
/// The backend receives:
///   - block      : sc_low.* IR, fully shaped, all high-level ops eliminated
///   - allocation : Value* -> BufferSlot map from BufferAllocator
///   - weights    : WeightBuffer containing all initializer data
class ICodegenTarget {
public:
    virtual ~ICodegenTarget() = default;

    /// Human-readable target name for Logger output.
    virtual std::string_view name() const = 0;

    /// Emit the output artifact (C source, CUDA PTX, etc.).
    /// Returns BackendError on failure — never throws.
    [[nodiscard]]
    virtual std::expected<void, BackendError> emit(
        const sir::Block&           block,
        const BufferAllocation&     allocation,
        const utility::WeightBuffer& weights) = 0;
};

// =============================================================================
// CodegenDriver — orchestrates the full codegen pipeline
// =============================================================================

/// Runs the three codegen stages in order:
///   WeightFolding -> BufferAllocation -> ICodegenTarget::emit
///
/// Usage:
///   CodegenDriver driver(std::make_unique<CpuCCodegenTarget>("out/model.c"));
///   auto result = driver.run(block, weights);
class CodegenDriver {
public:
    explicit CodegenDriver(std::unique_ptr<ICodegenTarget> target)
        : target_(std::move(target)) {}

    [[nodiscard]]
    std::expected<void, BackendError>
    run(sir::Block& block, utility::WeightBuffer& weights);

    /// Byte count of the activation arena from the most recent run().
    size_t arenaBytes() const { return arena_bytes_; }

private:
    std::unique_ptr<ICodegenTarget> target_;
    size_t                          arena_bytes_ = 0;
};

} // namespace seecpp::backend