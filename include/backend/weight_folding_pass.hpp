#pragma once

#include "backend/codegen.hpp"
#include "middle-end/sir.hpp"
#include "utility/weight_buffer.hpp"

#include <expected>
#include <string_view>

namespace seecpp::backend {

class WeightFoldingPass {
public:
    WeightFoldingPass() = default;

    std::string_view name() const { return "WeightFoldingPass"; }

    /// Run weight folding over `block`.
    /// Reads from and writes to `weights`.
    [[nodiscard]]
    std::expected<void, CodegenError>
    run(sir::Block& block, utility::WeightBuffer& weights);

private:
    /// Fold BN parameters into the filter and bias of one Conv op.
    [[nodiscard]]
    std::expected<void, CodegenError>
    foldConv(sir::Operation* conv_op, utility::WeightBuffer& weights);

    int folded_count_ = 0;
};

} // namespace seecpp::backend
