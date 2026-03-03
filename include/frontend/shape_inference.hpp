#pragma once

#include "middle-end/sir.hpp"
#include "utility/logger.hpp"

#include <string>
#include <expected>
#include <optional>
#include <unordered_map>
#include <functional>

namespace seecpp::frontend { 

// =============================================================================
// Diagnostics
// =============================================================================

struct ShapeError {
    std::string op_mnemonic;   // which op triggered the failure
    std::string value_id;      // which result Value was unresolvable
    std::string message;
};

// =============================================================================
// Shape utilities — free functions, reusable across passes
// =============================================================================

namespace shape_utils {

    /// NumPy / ONNX broadcast shape for two operands.
    /// Returns nullopt if the shapes are not broadcast-compatible.
    std::optional<sir::Shape> inferBroadcastShape(
        const sir::Shape& a, const sir::Shape& b);

    /// Convolution output spatial dimension formula:
    ///   floor((in + pad_begin + pad_end - dilation*(kernel-1) - 1) / stride + 1)
    int64_t convOutputDim(
        int64_t in_dim, int64_t kernel, int64_t stride,
        int64_t pad_begin, int64_t pad_end, int64_t dilation);

    /// Returns true iff every dimension in the shape is statically known.
    bool isFullyResolved(const sir::Shape& shape);

} // namespace shape_utils

// =============================================================================
// ShapeInferencePass
// =============================================================================

/// Performs a single forward pass over a Block, resolving all placeholder
/// Shape{} values left by the ingressor.
///
/// - Dynamic dimensions (kDynamic) are propagated, not treated as errors.
/// - All operand shapes must be known before an op's output can be inferred.
/// - Unknown ops that were emitted as sc_high.unknown passthrough are warned
///   about but do not abort the pass.
///
/// Renamed from ShapeInferenceEngine: "Pass" is the standard term in MLIR,
/// TVM Relay, and XLA for a single-traversal IR transformation.
class ShapeInferencePass {
public:
    ShapeInferencePass() = default;

    /// Run shape inference over `block`. Returns ShapeError on the first
    /// op whose output shape cannot be resolved.
    [[nodiscard]]
    std::expected<void, ShapeError> run(sir::Block& block);

private:
    // --- Per-op inference handlers ---
    // Each returns the inferred Shape for result(0), or ShapeError on failure.
    using InferFn = std::function
        std::expected<void, ShapeError>(sir::Operation*)>;

    [[nodiscard]] std::expected<void, ShapeError> inferMatMul(sir::Operation* op);
    [[nodiscard]] std::expected<void, ShapeError> inferConv2D(sir::Operation* op);
    [[nodiscard]] std::expected<void, ShapeError> inferElementwise(sir::Operation* op);
    [[nodiscard]] std::expected<void, ShapeError> inferReshape(sir::Operation* op);
    [[nodiscard]] std::expected<void, ShapeError> inferPooling(sir::Operation* op);
    [[nodiscard]] std::expected<void, ShapeError> inferBatchNorm(sir::Operation* op);
    [[nodiscard]] std::expected<void, ShapeError> inferConcat(sir::Operation* op);
    [[nodiscard]] std::expected<void, ShapeError> inferTranspose(sir::Operation* op);
    [[nodiscard]] std::expected<void, ShapeError> inferGemm(sir::Operation* op);

    // --- Validation helpers ---
    /// Checks all operands of `op` have a non-empty, known shape.
    [[nodiscard]]
    static std::expected<void, ShapeError>
    verifyOperandsResolved(sir::Operation* op);

    /// After the full pass, confirm no Shape{} placeholders remain.
    [[nodiscard]]
    std::expected<void, ShapeError>
    verifyAllShapesResolved(sir::Block& block) const;

    // Dispatch table: mnemonic suffix -> handler.
    // e.g. "sc_high.conv2d" -> inferConv2D
    static const std::unordered_map<std::string, InferFn> kInferFns;
};

} // namespace seecpp::middle