#include "include/backend/weight_folding_pass.hpp"
#include "utility/logger.hpp"

#include <cmath>
#include <vector>
#include <numeric>

namespace seecpp::backend {

std::expected<void, CodegenError>
WeightFoldingPass::run(sir::Block& block, utility::WeightBuffer& weights) {

    folded_count_ = 0;

    std::vector<sir::Operation*> fused_convs;
    block.walk([&](sir::Operation* op) {
        if (op->mnemonic() == "sc_high.conv2d" &&
            op->getAttrAs<int64_t>("fused_bn").value_or(0) == 1)
            fused_convs.push_back(op);
    });

    utility::Logger::info(
        "WeightFoldingPass: " + std::to_string(fused_convs.size()) +
        " fused Conv+BN op(s) to fold");

    for (auto* conv_op : fused_convs) {
        if (auto res = foldConv(conv_op, weights); !res)
            return res;
    }

    utility::Logger::info(
        "WeightFoldingPass: folded " +
        std::to_string(folded_count_) + " Conv+BN weight pair(s)");

    return {};
}

std::expected<void, CodegenError>
WeightFoldingPass::foldConv(sir::Operation*       conv_op,
                             utility::WeightBuffer& weights)
{
    // --- 1. Recover BN parameter ids from fusion attributes ---
    auto scale_id   = conv_op->getAttrAs<std::string>("bn_scale_id");
    auto bias_id    = conv_op->getAttrAs<std::string>("bn_bias_id");
    auto mean_id    = conv_op->getAttrAs<std::string>("bn_mean_id");
    auto var_id     = conv_op->getAttrAs<std::string>("bn_var_id");
    float eps       = conv_op->getAttrAs<float>("bn_epsilon").value_or(1e-5f);

    if (!scale_id || !bias_id || !mean_id || !var_id)
        return std::unexpected(CodegenError{
            "weight_folding",
            "Conv op marked fused_bn=1 is missing BN parameter id attributes"});

    // --- 2. Fetch BN tensors from WeightBuffer ---
    auto scale_span = weights.get<float>(*scale_id);
    auto bn_bias_sp = weights.get<float>(*bias_id);
    auto mean_span  = weights.get<float>(*mean_id);
    auto var_span   = weights.get<float>(*var_id);

    if (!scale_span || !bn_bias_sp || !mean_span || !var_span)
        return std::unexpected(CodegenError{
            "weight_folding",
            "Could not retrieve BN tensors from WeightBuffer for Conv op. "
            "Ensure the ONNX ingressor populated all four BN initializers."});

    const size_t F = scale_span->size();   // number of output channels

    if (bn_bias_sp->size() != F || mean_span->size() != F || var_span->size() != F)
        return std::unexpected(CodegenError{
            "weight_folding",
            "BN parameter size mismatch — all four tensors must have F elements "
            "(one per output channel)"});

    // --- 3. Fetch Conv filter ---
    // Filter operand id: the SSA id of the weight Value (operand 1).
    if (conv_op->numOperands() < 2)
        return std::unexpected(CodegenError{
            "weight_folding",
            "Conv op has fewer than 2 operands — cannot identify filter"});

    const std::string filter_id(conv_op->operand(1)->id());
    auto filter_span = weights.get<float>(filter_id);

    if (!filter_span)
        return std::unexpected(CodegenError{
            "weight_folding",
            "Conv filter '" + filter_id + "' not found in WeightBuffer"});

    // Filter layout: [F, C/group, KH, KW] — F is outermost dimension.
    const size_t filter_total  = filter_span->size();
    const size_t elems_per_out = (F > 0) ? filter_total / F : 0;

    if (elems_per_out * F != filter_total)
        return std::unexpected(CodegenError{
            "weight_folding",
            "Filter element count " + std::to_string(filter_total) +
            " is not divisible by F=" + std::to_string(F)});

    // --- 4. Fetch or synthesise Conv bias ---
    std::vector<float> bias_data(F, 0.0f);

    const bool has_bias = conv_op->numOperands() >= 3;
    if (has_bias) {
        const std::string bias_id_str(conv_op->operand(2)->id());
        auto bias_span = weights.get<float>(bias_id_str);
        if (bias_span) {
            if (bias_span->size() != F)
                return std::unexpected(CodegenError{
                    "weight_folding",
                    "Conv bias size " + std::to_string(bias_span->size()) +
                    " does not match F=" + std::to_string(F)});
            std::copy(bias_span->begin(), bias_span->end(), bias_data.begin());
        }
    }

    // --- 5. Compute folded weights ---
    //
    //   std_inv[f] = 1 / sqrt(var[f] + eps)
    //   scale_factor[f] = gamma[f] * std_inv[f]
    //
    //   W_fused[f, :] = W[f, :] * scale_factor[f]
    //   b_fused[f]    = (b[f] - mean[f]) * scale_factor[f] + beta[f]

    std::vector<float> w_fused(filter_total);
    std::vector<float> b_fused(F);

    for (size_t f = 0; f < F; ++f) {
        const float std_inv      = 1.0f / std::sqrt((*var_span)[f] + eps);
        const float scale_factor = (*scale_span)[f] * std_inv;

        // Scale every filter element in output channel f.
        const float* src = filter_span->data() + f * elems_per_out;
        float*       dst = w_fused.data()       + f * elems_per_out;
        for (size_t i = 0; i < elems_per_out; ++i)
            dst[i] = src[i] * scale_factor;

        b_fused[f] = (bias_data[f] - (*mean_span)[f]) * scale_factor
                     + (*bn_bias_sp)[f];
    }

    // --- 6. Write folded tensors back to WeightBuffer ---
    const std::string w_folded_key = filter_id + "__bn_folded";
    const std::string b_folded_key = filter_id + "__bn_folded_bias";

    weights.add<float>(w_folded_key,
                        std::span<const float>(w_fused),
                        utility::WeightBuffer::BufferDtype::F32);
    weights.add<float>(b_folded_key,
                        std::span<const float>(b_fused),
                        utility::WeightBuffer::BufferDtype::F32);

    // --- 7. Update Conv op attributes so codegen reads the folded tensors ---
    conv_op->setAttribute("folded_weight_key", w_folded_key);
    conv_op->setAttribute("folded_bias_key",   b_folded_key);

    // Clear fused_bn so this op is not re-processed on a second driver run.
    conv_op->setAttribute("fused_bn", int64_t(0));

    ++folded_count_;

    utility::Logger::info(
        "WeightFoldingPass: folded '" + filter_id + "' -> '" + w_folded_key +
        "' [F=" + std::to_string(F) +
        " elems_per_channel=" + std::to_string(elems_per_out) + "]");

    return {};
}

} // namespace seecpp::backend