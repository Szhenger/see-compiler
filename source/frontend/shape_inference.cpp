#include "include/frontend/shape_inference.hpp"

#include <cmath>
#include <algorithm>
#include <cassert>

namespace seecpp::frontend {

// =============================================================================
// shape_utils — free functions
// =============================================================================

namespace shape_utils {

std::optional<sir::Shape> inferBroadcastShape(
    const sir::Shape& a, const sir::Shape& b)
{
    // NumPy broadcast rules: right-align dims, then for each pair:
    //   - equal        -> keep
    //   - one is 1     -> take the other
    //   - one is kDyn  -> kDynamic (we don't know at compile time)
    //   - incompatible -> nullopt

    const size_t rank = std::max(a.dims.size(), b.dims.size());
    sir::Shape   out;
    out.dims.resize(rank);

    for (size_t i = 0; i < rank; ++i) {
        // Right-align: pad the shorter shape with implicit 1s on the left.
        int64_t da = (i < rank - a.dims.size())
                         ? 1
                         : a.dims[i - (rank - a.dims.size())];
        int64_t db = (i < rank - b.dims.size())
                         ? 1
                         : b.dims[i - (rank - b.dims.size())];

        if (da == sir::Shape::kDynamic || db == sir::Shape::kDynamic) {
            out.dims[i] = sir::Shape::kDynamic;
        } else if (da == db) {
            out.dims[i] = da;
        } else if (da == 1) {
            out.dims[i] = db;
        } else if (db == 1) {
            out.dims[i] = da;
        } else {
            return std::nullopt;  // broadcast-incompatible
        }
    }
    return out;
}

int64_t convOutputDim(
    int64_t in_dim,   int64_t kernel,
    int64_t stride,   int64_t pad_begin,
    int64_t pad_end,  int64_t dilation)
{
    if (in_dim == sir::Shape::kDynamic) return sir::Shape::kDynamic;
    return (in_dim + pad_begin + pad_end - dilation * (kernel - 1) - 1) / stride + 1;
}

bool isFullyResolved(const sir::Shape& shape) {
    if (shape.dims.empty()) return false;  // placeholder Shape{}
    for (auto d : shape.dims)
        if (d == sir::Shape::kDynamic) return false;
    return true;
}

} // namespace shape_utils

// =============================================================================
// Dispatch table
// =============================================================================
// Populated with member-function wrappers so each handler is a clean closure.
// Add a new op by appending one entry — no changes to run() required.
// =============================================================================

const std::unordered_map<std::string, ShapeInferencePass::InferFn>
ShapeInferencePass::kInferFns = {

    {"sc_high.matmul",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferMatMul(op); }},

    {"sc_high.gemm",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferGemm(op); }},

    {"sc_high.conv2d",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferConv2D(op); }},

    {"sc_high.relu",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferElementwise(op); }},
    {"sc_high.add",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferElementwise(op); }},
    {"sc_high.sub",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferElementwise(op); }},
    {"sc_high.mul",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferElementwise(op); }},
    {"sc_high.div",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferElementwise(op); }},

    {"sc_high.batch_norm",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferBatchNorm(op); }},

    {"sc_high.reshape",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferReshape(op); }},

    {"sc_high.maxpool",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferPooling(op); }},
    {"sc_high.avgpool",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferPooling(op); }},

    {"sc_high.concat",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferConcat(op); }},

    {"sc_high.transpose",
     [](sir::Operation* op, ShapeInferencePass* self) {
         return self->inferTranspose(op); }},

    // Ingressor-produced ops: shapes already set, nothing to infer.
    {"sc_high.constant",
     [](sir::Operation*, ShapeInferencePass*) -> std::expected<void, ShapeError> {
         return {}; }},
};

// =============================================================================
// Public entry point
// =============================================================================

[[nodiscard]]
std::expected<void, ShapeError> ShapeInferencePass::run(sir::Block& block) {

    utility::Logger::info("ShapeInferencePass: starting forward pass...");

    // Pre-condition: block must pass structural validation before we touch shapes.
    if (!block.validate())
        return std::unexpected(ShapeError{
            "", "", "Block failed structural validation before shape inference"});

    for (const auto& owned_op : block.operations()) {
        sir::Operation* op = owned_op.get();
        const std::string mnemonic(op->mnemonic());

        // Block arguments and sc_high.unknown passthrough ops have no results
        // to infer, but we warn loudly on unknowns so they aren't silently lost.
        if (mnemonic == "sc_high.unknown") {
            utility::Logger::warn(
                "ShapeInferencePass: skipping unrecognised op '" +
                std::string(op->getAttrAs<std::string>("onnx_op_type")
                                .value_or("?")) +
                "' — output shapes remain dynamic");
            continue;
        }

        auto it = kInferFns.find(mnemonic);
        if (it == kInferFns.end()) {
            return std::unexpected(ShapeError{
                mnemonic, "",
                "No shape inference handler registered for '" + mnemonic + "'"});
        }

        // Verify all input shapes are resolved before attempting inference.
        if (auto chk = verifyOperandsResolved(op); !chk)
            return std::unexpected(chk.error());

        if (auto res = it->second(op, this); !res)
            return std::unexpected(res.error());
    }

    // Post-condition: every result Value must now have a concrete shape.
    if (auto chk = verifyAllShapesResolved(block); !chk)
        return std::unexpected(chk.error());

    utility::Logger::info("ShapeInferencePass: all shapes resolved successfully.");
    return {};
}

// =============================================================================
// Validation helpers
// =============================================================================

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::verifyOperandsResolved(sir::Operation* op) {
    for (const auto* operand : op->operands()) {
        if (operand->shape().dims.empty()) {
            return std::unexpected(ShapeError{
                std::string(op->mnemonic()),
                std::string(operand->id()),
                "Operand '" + std::string(operand->id()) +
                    "' has an unresolved shape before inference"});
        }
    }
    return {};
}

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::verifyAllShapesResolved(sir::Block& block) const {
    for (const auto& owned_op : block.operations()) {
        for (size_t i = 0; i < owned_op->numResults(); ++i) {
            const sir::Value* v = owned_op->result(i);
            if (v->shape().dims.empty()) {
                return std::unexpected(ShapeError{
                    std::string(owned_op->mnemonic()),
                    std::string(v->id()),
                    "Result '" + std::string(v->id()) +
                        "' still has an unresolved shape after inference pass"});
            }
        }
    }
    return {};
}

// =============================================================================
// Per-op inference handlers
// =============================================================================

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::inferMatMul(sir::Operation* op) {
    // Operands: [A, B]
    // Rule (ONNX MatMul): batch dims broadcast; inner dims contract.
    // [*, M, K] x [*, K, N] -> [*, M, N]

    if (op->numOperands() < 2)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "",
            "MatMul requires exactly 2 operands"});

    const auto& dimsA = op->operand(0)->shape().dims;
    const auto& dimsB = op->operand(1)->shape().dims;

    if (dimsA.size() < 2 || dimsB.size() < 2)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "",
            "MatMul operands must be rank >= 2"});

    const int64_t K_a = dimsA.back();
    const int64_t K_b = dimsB[dimsB.size() - 2];

    if (K_a != sir::Shape::kDynamic &&
        K_b != sir::Shape::kDynamic &&
        K_a != K_b)
    {
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "",
            "MatMul: inner dimensions mismatch (" +
                std::to_string(K_a) + " vs " + std::to_string(K_b) + ")"});
    }

    // Broadcast the batch dimensions (everything except the last two).
    sir::Shape batchA, batchB;
    batchA.dims.assign(dimsA.begin(), dimsA.end() - 2);
    batchB.dims.assign(dimsB.begin(), dimsB.end() - 2);

    sir::Shape outBatch;
    if (!batchA.dims.empty() || !batchB.dims.empty()) {
        auto b = shape_utils::inferBroadcastShape(batchA, batchB);
        if (!b)
            return std::unexpected(ShapeError{
                std::string(op->mnemonic()), "",
                "MatMul: batch dimensions are not broadcast-compatible"});
        outBatch = *b;
    }

    outBatch.dims.push_back(dimsA[dimsA.size() - 2]);  // M
    outBatch.dims.push_back(dimsB.back());              // N

    op->result(0)->setShape(std::move(outBatch));
    return {};
}

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::inferGemm(sir::Operation* op) {
    // ONNX Gemm: C = alpha * A' * B' + beta * bias
    // A: [M, K], B: [K, N] (after optional transpose) -> output: [M, N]

    if (op->numOperands() < 2)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "", "Gemm requires at least 2 operands"});

    const auto& dimsA = op->operand(0)->shape().dims;
    const auto& dimsB = op->operand(1)->shape().dims;

    if (dimsA.size() != 2 || dimsB.size() != 2)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "", "Gemm operands must be rank-2"});

    bool trans_a = op->getAttrAs<int64_t>("trans_a").value_or(0) != 0;
    bool trans_b = op->getAttrAs<int64_t>("trans_b").value_or(0) != 0;

    int64_t M = trans_a ? dimsA[1] : dimsA[0];
    int64_t N = trans_b ? dimsB[0] : dimsB[1];

    op->result(0)->setShape(sir::Shape{{M, N}});
    return {};
}

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::inferConv2D(sir::Operation* op) {
    // Operands: [input, filter] (bias optional)
    // input  : [N, C_in, H, W]
    // filter : [C_out, C_in/group, KH, KW]
    // output : [N, C_out, out_H, out_W]

    if (op->numOperands() < 2)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "", "Conv2D requires at least 2 operands"});

    const auto& in  = op->operand(0)->shape().dims;
    const auto& fil = op->operand(1)->shape().dims;

    if (in.size() != 4 || fil.size() != 4)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "",
            "Conv2D: input and filter must be rank-4 (NCHW)"});

    // Retrieve attributes with safe defaults.
    auto strides   = op->getAttrAs<std::vector<int64_t>>("strides")
                         .value_or(std::vector<int64_t>{1, 1});
    auto pads      = op->getAttrAs<std::vector<int64_t>>("pads")
                         .value_or(std::vector<int64_t>{0, 0, 0, 0});
    auto dilations = op->getAttrAs<std::vector<int64_t>>("dilations")
                         .value_or(std::vector<int64_t>{1, 1});

    // pads layout: [top, left, bottom, right]
    int64_t out_h = shape_utils::convOutputDim(
        in[2], fil[2], strides[0], pads[0], pads[2], dilations[0]);
    int64_t out_w = shape_utils::convOutputDim(
        in[3], fil[3], strides[1], pads[1], pads[3], dilations[1]);

    op->result(0)->setShape(sir::Shape{{in[0], fil[0], out_h, out_w}});
    return {};
}

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::inferElementwise(sir::Operation* op) {
    // Unary ops (Relu, Sigmoid, etc.): output shape == input shape.
    if (op->numOperands() == 1) {
        op->result(0)->setShape(op->operand(0)->shape());
        return {};
    }

    // Binary ops (Add, Sub, Mul, Div): NumPy broadcast.
    if (op->numOperands() == 2) {
        auto out = shape_utils::inferBroadcastShape(
            op->operand(0)->shape(), op->operand(1)->shape());
        if (!out)
            return std::unexpected(ShapeError{
                std::string(op->mnemonic()), "",
                "Elementwise op: operand shapes are not broadcast-compatible"});
        op->result(0)->setShape(*out);
        return {};
    }

    return std::unexpected(ShapeError{
        std::string(op->mnemonic()), "",
        "Elementwise op: expected 1 or 2 operands"});
}

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::inferBatchNorm(sir::Operation* op) {
    // BatchNorm is shape-preserving: output == input.
    if (op->numOperands() < 1)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "", "BatchNorm: missing input operand"});

    op->result(0)->setShape(op->operand(0)->shape());
    return {};
}

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::inferReshape(sir::Operation* op) {
    // Operands: [input, shape_tensor]
    // The target shape is stored as an attribute "target_shape" by the ingressor.
    auto target = op->getAttrAs<std::vector<int64_t>>("target_shape");
    if (!target)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "",
            "Reshape: missing 'target_shape' attribute"});

    // Resolve a single -1 (infer dimension) from the input volume.
    int64_t known_vol = 1;
    int infer_idx     = -1;

    for (size_t i = 0; i < target->size(); ++i) {
        if ((*target)[i] == -1) {
            if (infer_idx != -1)
                return std::unexpected(ShapeError{
                    std::string(op->mnemonic()), "",
                    "Reshape: at most one dimension may be -1"});
            infer_idx = static_cast<int>(i);
        } else {
            known_vol *= (*target)[i];
        }
    }

    if (infer_idx != -1) {
        int64_t input_vol = op->operand(0)->shape().volume();
        if (input_vol == sir::Shape::kDynamic) {
            (*target)[infer_idx] = sir::Shape::kDynamic;
        } else {
            (*target)[infer_idx] = input_vol / known_vol;
        }
    }

    op->result(0)->setShape(sir::Shape{*target});
    return {};
}

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::inferPooling(sir::Operation* op) {
    // MaxPool / AveragePool: same formula as Conv2D but no filter-count dim.
    // input: [N, C, H, W] -> output: [N, C, out_H, out_W]

    if (op->numOperands() < 1)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "", "Pooling: missing input operand"});

    const auto& in = op->operand(0)->shape().dims;
    if (in.size() != 4)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "",
            "Pooling: input must be rank-4 (NCHW)"});

    auto kernel    = op->getAttrAs<std::vector<int64_t>>("kernel_shape")
                         .value_or(std::vector<int64_t>{1, 1});
    auto strides   = op->getAttrAs<std::vector<int64_t>>("strides")
                         .value_or(std::vector<int64_t>{1, 1});
    auto pads      = op->getAttrAs<std::vector<int64_t>>("pads")
                         .value_or(std::vector<int64_t>{0, 0, 0, 0});
    auto dilations = op->getAttrAs<std::vector<int64_t>>("dilations")
                         .value_or(std::vector<int64_t>{1, 1});

    int64_t out_h = shape_utils::convOutputDim(
        in[2], kernel[0], strides[0], pads[0], pads[2], dilations[0]);
    int64_t out_w = shape_utils::convOutputDim(
        in[3], kernel[1], strides[1], pads[1], pads[3], dilations[1]);

    op->result(0)->setShape(sir::Shape{{in[0], in[1], out_h, out_w}});
    return {};
}

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::inferConcat(sir::Operation* op) {
    // ONNX Concat: stack N tensors along `axis`.
    // All dims must match except on the concat axis.

    auto axis_opt = op->getAttrAs<int64_t>("axis");
    if (!axis_opt)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "", "Concat: missing 'axis' attribute"});

    if (op->numOperands() == 0)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "", "Concat: no operands"});

    const auto& base_dims = op->operand(0)->shape().dims;
    int64_t rank          = static_cast<int64_t>(base_dims.size());
    int64_t axis          = *axis_opt < 0 ? *axis_opt + rank : *axis_opt;

    if (axis < 0 || axis >= rank)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "",
            "Concat: axis " + std::to_string(axis) + " out of range for rank " +
                std::to_string(rank)});

    std::vector<int64_t> out_dims = base_dims;

    for (size_t i = 1; i < op->numOperands(); ++i) {
        const auto& d = op->operand(i)->shape().dims;
        if ((int64_t)d.size() != rank)
            return std::unexpected(ShapeError{
                std::string(op->mnemonic()), "",
                "Concat: operand " + std::to_string(i) + " has mismatched rank"});

        for (int64_t j = 0; j < rank; ++j) {
            if (j == axis) {
                out_dims[j] = (out_dims[j] == sir::Shape::kDynamic ||
                               d[j]         == sir::Shape::kDynamic)
                                  ? sir::Shape::kDynamic
                                  : out_dims[j] + d[j];
            } else if (out_dims[j] != d[j]) {
                return std::unexpected(ShapeError{
                    std::string(op->mnemonic()), "",
                    "Concat: non-axis dimensions mismatch at dim " +
                        std::to_string(j)});
            }
        }
    }

    op->result(0)->setShape(sir::Shape{out_dims});
    return {};
}

[[nodiscard]]
std::expected<void, ShapeError>
ShapeInferencePass::inferTranspose(sir::Operation* op) {
    // ONNX Transpose: permutes dims according to `perm` attribute.
    // Default perm (absent attribute) reverses all axes.

    if (op->numOperands() < 1)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "", "Transpose: missing input operand"});

    const auto& in_dims = op->operand(0)->shape().dims;
    const int64_t rank  = static_cast<int64_t>(in_dims.size());

    std::vector<int64_t> perm;
    if (auto p = op->getAttrAs<std::vector<int64_t>>("perm")) {
        perm = *p;
    } else {
        // Default: reverse
        perm.resize(rank);
        std::iota(perm.rbegin(), perm.rend(), 0);
    }

    if ((int64_t)perm.size() != rank)
        return std::unexpected(ShapeError{
            std::string(op->mnemonic()), "",
            "Transpose: perm length does not match input rank"});

    std::vector<int64_t> out_dims(rank);
    for (int64_t i = 0; i < rank; ++i)
        out_dims[i] = in_dims[perm[i]];

    op->result(0)->setShape(sir::Shape{out_dims});
    return {};
}

} // namespace seecpp::middle