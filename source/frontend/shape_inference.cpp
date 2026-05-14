#include "include/frontend/shape_inference.hpp"
#include "include/utility/logger.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace seecpp::frontend {

// =============================================================================
// shape_utils
// =============================================================================

namespace shape_utils {

std::optional<sir::Shape> inferBroadcastShape(const sir::Shape& a, const sir::Shape& b) {
    const size_t rank = std::max(a.dims.size(), b.dims.size());
    sir::Shape out;
    out.dims.resize(rank);

    for (size_t i = 0; i < rank; ++i) {
        int64_t da = (i < rank - a.dims.size()) ? 1 : a.dims[i - (rank - a.dims.size())];
        int64_t db = (i < rank - b.dims.size()) ? 1 : b.dims[i - (rank - b.dims.size())];

        if (da == sir::Shape::kDynamic || db == sir::Shape::kDynamic) {
            out.dims[i] = sir::Shape::kDynamic;
        } else if (da == db || da == 1 || db == 1) {
            out.dims[i] = std::max(da, db);
        } else {
            return std::nullopt;
        }
    }
    return out;
}

int64_t convOutputDim(int64_t in_dim, int64_t kernel, int64_t stride, 
                      int64_t pad_begin, int64_t pad_end, int64_t dilation) {
    if (in_dim == sir::Shape::kDynamic) return sir::Shape::kDynamic;
    
    // Effective kernel size considering dilation
    int64_t effective_k = (kernel - 1) * dilation + 1;
    int64_t padded_in = in_dim + pad_begin + pad_end;
    
    if (padded_in < effective_k) return 0; // Invalid math resulting in 0-size or negative
    
    return (padded_in - effective_k) / stride + 1;
}

bool isFullyResolved(const sir::Shape& shape) {
    if (shape.dims.empty()) return false;
    return std::none_of(shape.dims.begin(), shape.dims.end(), 
                        [](int64_t d) { return d == sir::Shape::kDynamic; });
}

} // namespace shape_utils

// =============================================================================
// Dispatch Table (Transparent Lookups)
// =============================================================================

const std::unordered_map<std::string, ShapeInferencePass::InferFn, 
                         ShapeInferencePass::StringHash, std::equal_to<>>
ShapeInferencePass::kInferFns = {
    {"sc_high.matmul",      &ShapeInferencePass::inferMatMul},
    {"sc_high.gemm",        &ShapeInferencePass::inferGemm},
    {"sc_high.conv2d",      &ShapeInferencePass::inferConv2D},
    {"sc_high.relu",        &ShapeInferencePass::inferElementwise},
    {"sc_high.add",         &ShapeInferencePass::inferElementwise},
    {"sc_high.sub",         &ShapeInferencePass::inferElementwise},
    {"sc_high.mul",         &ShapeInferencePass::inferElementwise},
    {"sc_high.div",         &ShapeInferencePass::inferElementwise},
    {"sc_high.batch_norm",  &ShapeInferencePass::inferBatchNorm},
    {"sc_high.reshape",     &ShapeInferencePass::inferReshape},
    {"sc_high.maxpool",     &ShapeInferencePass::inferPooling},
    {"sc_high.avgpool",     &ShapeInferencePass::inferPooling},
    {"sc_high.concat",      &ShapeInferencePass::inferConcat},
    {"sc_high.transpose",   &ShapeInferencePass::inferTranspose},
    {"sc_high.constant",    [](sir::Operation*) -> std::expected<void, ShapeError> { return {}; }}
};

// =============================================================================
// Public Entry Point
// =============================================================================

std::expected<void, ShapeError> ShapeInferencePass::run(sir::Block& block) {
    utility::Logger::info("ShapeInferencePass: executing forward propagation...");

    for (const auto& owned_op : block.operations()) {
        sir::Operation* op = owned_op.get();
        std::string_view mnemonic = op->mnemonic();

        if (mnemonic == "sc_high.unknown") continue;

        auto it = kInferFns.find(mnemonic);
        if (it == kInferFns.end()) {
            return std::unexpected(ShapeError{
                ShapeErrorCode::InternalError, std::string(mnemonic),
                "Missing inference handler for op"
            });
        }

        if (auto chk = verifyOperandsResolved(op); !chk) return chk;

        // Use 'this' to call the member function pointer from the map
        if (auto res = std::invoke(it->second, this, op); !res) return res;
    }

    return verifyAllShapesResolved(block);
}

// =============================================================================
// Refined Handlers
// =============================================================================

std::expected<void, ShapeError> ShapeInferencePass::inferConv2D(sir::Operation* op) {
    if (op->numOperands() < 2) {
        return std::unexpected(ShapeError{ShapeErrorCode::UnresolvedInput, 
                               std::string(op->mnemonic()), "Insufficient operands"});
    }

    const auto& in = op->operand(0)->shape().dims;
    const auto& fil = op->operand(1)->shape().dims;

    if (in.size() != 4 || fil.size() != 4) {
        return std::unexpected(ShapeError{ShapeErrorCode::IncompatibleOperands, 
                               std::string(op->mnemonic()), "Conv2D requires NCHW (rank-4)"});
    }

    auto strides   = op->getAttrAs<std::vector<int64_t>>("strides").value_or(std::vector<int64_t>{1, 1});
    auto pads      = op->getAttrAs<std::vector<int64_t>>("pads").value_or(std::vector<int64_t>{0, 0, 0, 0});
    auto dilations = op->getAttrAs<std::vector<int64_t>>("dilations").value_or(std::vector<int64_t>{1, 1});

    int64_t out_h = shape_utils::convOutputDim(in[2], fil[2], strides[0], pads[0], pads[2], dilations[0]);
    int64_t out_w = shape_utils::convOutputDim(in[3], fil[3], strides[1], pads[1], pads[3], dilations[1]);

    if (out_h == 0 || out_w == 0) {
        return std::unexpected(ShapeError{ShapeErrorCode::MathematicalViolation, 
                               std::string(op->mnemonic()), "Spatial dimensions collapsed to zero"});
    }

    op->result(0)->setShape(sir::Shape{{in[0], fil[0], out_h, out_w}});
    return {};
}

std::expected<void, ShapeError> ShapeInferencePass::inferReshape(sir::Operation* op) {
    auto target_opt = op->getAttrAs<std::vector<int64_t>>("target_shape");
    if (!target_opt) {
        return std::unexpected(ShapeError{ShapeErrorCode::InvalidAttribute, 
                               std::string(op->mnemonic()), "Missing target_shape"});
    }

    std::vector<int64_t> target = *target_opt;
    int64_t input_vol = op->operand(0)->shape().volume();
    int64_t known_vol = 1;
    int infer_idx = -1;

    for (size_t i = 0; i < target.size(); ++i) {
        if (target[i] == -1) {
            if (infer_idx != -1) return std::unexpected(ShapeError{ShapeErrorCode::MathematicalViolation, 
                                                        std::string(op->mnemonic()), "Multiple -1 dims"});
            infer_idx = static_cast<int>(i);
        } else {
            known_vol *= target[i];
        }
    }

    if (infer_idx != -1) {
        if (input_vol == sir::Shape::kDynamic) {
            target[infer_idx] = sir::Shape::kDynamic;
        } else {
            if (input_vol % known_vol != 0) {
                return std::unexpected(ShapeError{ShapeErrorCode::MathematicalViolation, 
                                       std::string(op->mnemonic()), "Incompatible reshape volume"});
            }
            target[infer_idx] = input_vol / known_vol;
        }
    }

    op->result(0)->setShape(sir::Shape{std::move(target)});
    return {};
}

std::expected<void, ShapeError> ShapeInferencePass::verifyOperandsResolved(sir::Operation* op) {
    for (const auto* operand : op->operands()) {
        if (operand->shape().dims.empty()) {
            return std::unexpected(ShapeError{
                ShapeErrorCode::UnresolvedInput, std::string(op->mnemonic()),
                "Input '" + std::string(operand->id()) + "' has no shape"
            });
        }
    }
    return {};
}

std::expected<void, ShapeError> ShapeInferencePass::verifyAllShapesResolved(sir::Block& block) const {
    for (const auto& owned_op : block.operations()) {
        for (size_t i = 0; i < owned_op->numResults(); ++i) {
            if (owned_op->result(i)->shape().dims.empty()) {
                return std::unexpected(ShapeError{
                    ShapeErrorCode::InternalError, std::string(owned_op->mnemonic()),
                    "Output resolution failed"
                });
            }
        }
    }
    return {};
}

} // namespace seecpp::frontend
