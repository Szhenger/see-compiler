#pragma once

#include "middle-end/sir.hpp"
#include "utility/logger.hpp"

#include <string>
#include <string_view>
#include <expected>
#include <optional>
#include <unordered_map>
#include <functional>

namespace seecpp::frontend { 

// =============================================================================
// Diagnostics
// =============================================================================

enum class ShapeErrorCode {
    IncompatibleOperands,
    UnresolvedInput,
    InvalidAttribute,
    MathematicalViolation,
    InternalError
};

struct ShapeError {
    ShapeErrorCode code;
    std::string op_mnemonic;   
    std::string message;
    sir::Value* target_value = nullptr; 
};

// =============================================================================
// Shape utilities
// =============================================================================

namespace shape_utils {
    std::optional<sir::Shape> inferBroadcastShape(const sir::Shape& a, const sir::Shape& b);

    int64_t convOutputDim(
        int64_t in_dim, int64_t kernel, int64_t stride,
        int64_t pad_begin, int64_t pad_end, int64_t dilation);

    bool isFullyResolved(const sir::Shape& shape);
} 

// =============================================================================
// ShapeInferencePass
// =============================================================================

class ShapeInferencePass {
public:
    ShapeInferencePass() = default;

    [[nodiscard]]
    std::expected<void, ShapeError> run(sir::Block& block);

private:
    struct StringHash {
        using is_transparent = void;
        size_t operator()(std::string_view sv) const {
            return std::hash<std::string_view>{}(sv);
        }
    };

    using InferFn = std::function<std::expected<void, ShapeError>(sir::Operation*)>;

    std::expected<void, ShapeError> inferMatMul(sir::Operation* op);
    std::expected<void, ShapeError> inferConv2D(sir::Operation* op);
    std::expected<void, ShapeError> inferElementwise(sir::Operation* op);
    std::expected<void, ShapeError> inferReshape(sir::Operation* op);
    std::expected<void, ShapeError> inferPooling(sir::Operation* op);
    std::expected<void, ShapeError> inferBatchNorm(sir::Operation* op);
    std::expected<void, ShapeError> inferGemm(sir::Operation* op);


    static std::expected<void, ShapeError> verifyOperandsResolved(sir::Operation* op);
    static const std::unordered_map<std::string, InferFn, StringHash, std::equal_to<>> kInferFns;
};

} // namespace seecpp::frontend
