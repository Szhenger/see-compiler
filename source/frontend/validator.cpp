#include "frontend/validator.hpp"
#include "include/utility/logger.hpp"

#include <unordered_set>
#include <unordered_map>
#include <numeric>

namespace seecpp::frontend {

// =============================================================================
// Static data
// =============================================================================

const std::unordered_set<std::string> Validator::kSupportedOps = {
    "sc_high.matmul",
    "sc_high.gemm",
    "sc_high.conv2d",
    "sc_high.relu",
    "sc_high.add",
    "sc_high.sub",
    "sc_high.mul",
    "sc_high.div",
    "sc_high.batch_norm",
    "sc_high.reshape",
    "sc_high.maxpool",
    "sc_high.avgpool",
    "sc_high.concat",
    "sc_high.transpose",
    "sc_high.constant",
};

// =============================================================================
// Public entry point
// =============================================================================

[[nodiscard]]
ValidationReport Validator::validate(const sir::Block& block) {
    ValidationReport report;

    utility::Logger::info("Validator: starting multi-pass validation...");

    // All five passes always run so the caller gets the full diagnostic set.
    checkSSALinks(block, report);
    checkTopologicalOrder(block, report);

    for (const auto& owned_op : block.operations()) {
        const sir::Operation& op = *owned_op;
        const std::string mnemonic(op.mnemonic());

        // Dialect gate: sc_high.unknown is an explicit hard error here.
        // Unknown ops must be resolved before the block is handed to the
        // middle-end. Emit a single error per unknown op and continue so
        // all unknowns are reported at once.
        if (kSupportedOps.find(mnemonic) == kSupportedOps.end()) {
            report.diagnostics.push_back({
                ValidationError::Severity::Error,
                mnemonic, "",
                "Unsupported op '" + mnemonic +
                    "' — resolve sc_high.unknown nodes before validation"});
            continue;  // skip further per-op checks for unknown ops
        }

        checkOpConstraints(op,       report);
        checkTypeConsistency(op,     report);
        checkShapeConsistency(op,    report);
    }

    if (report.hasErrors()) {
        utility::Logger::error(
            "Validator: " +
            std::to_string(report.diagnostics.size()) +
            " issue(s) found.");
    } else {
        utility::Logger::info("Validator: all passes clean.");
    }

    return report;
}

// =============================================================================
// Pass 1 — SSA structural integrity
// =============================================================================

void Validator::checkSSALinks(
    const sir::Block& block, ValidationReport& report) const
{
    for (const auto& owned_op : block.operations()) {
        const sir::Operation* op = owned_op.get();

        // Every operand pointer must be non-null.
        for (size_t i = 0; i < op->numOperands(); ++i) {
            if (op->operand(i) == nullptr) {
                report.diagnostics.push_back({
                    ValidationError::Severity::Error,
                    std::string(op->mnemonic()), "",
                    "Operand " + std::to_string(i) + " is a null pointer"});
            }
        }

        // Every result pointer must be non-null and have a non-empty id.
        for (size_t i = 0; i < op->numResults(); ++i) {
            const sir::Value* res = op->result(i);
            if (!res) {
                report.diagnostics.push_back({
                    ValidationError::Severity::Error,
                    std::string(op->mnemonic()), "",
                    "Result " + std::to_string(i) + " is a null pointer"});
            } else if (res->id().empty()) {
                report.diagnostics.push_back({
                    ValidationError::Severity::Error,
                    std::string(op->mnemonic()), "",
                    "Result " + std::to_string(i) + " has an empty SSA id"});
            }
        }

        // Each operand's defining_op (or block-argument status) must be
        // consistent: non-block-arguments must have a non-null defining op.
        for (size_t i = 0; i < op->numOperands(); ++i) {
            const sir::Value* v = op->operand(i);
            if (!v) continue;  // already reported above
            if (!v->isBlockArgument() && v->definingOp() == nullptr) {
                report.diagnostics.push_back({
                    ValidationError::Severity::Error,
                    std::string(op->mnemonic()),
                    std::string(v->id()),
                    "Operand '" + std::string(v->id()) +
                        "' is not a block argument but has no defining op"});
            }
        }
    }
}

// =============================================================================
// Pass 2 — Topological order (DAG / use-before-def check)
// =============================================================================

void Validator::checkTopologicalOrder(
    const sir::Block& block, ValidationReport& report) const
{
    // Seed with block arguments — they are defined at block entry.
    std::unordered_set<const sir::Value*> defined;
    for (const auto& arg : block.arguments())
        defined.insert(arg.get());

    for (const auto& owned_op : block.operations()) {
        const sir::Operation* op = owned_op.get();

        for (size_t i = 0; i < op->numOperands(); ++i) {
            const sir::Value* v = op->operand(i);
            if (!v) continue;  // null already reported by checkSSALinks
            if (defined.find(v) == defined.end()) {
                report.diagnostics.push_back({
                    ValidationError::Severity::Error,
                    std::string(op->mnemonic()),
                    std::string(v->id()),
                    "SSA violation: '" + std::string(v->id()) +
                        "' used before definition"});
            }
        }

        // Mark this op's results as defined for subsequent ops.
        for (size_t i = 0; i < op->numResults(); ++i)
            if (op->result(i)) defined.insert(op->result(i));
    }
}

// =============================================================================
// Pass 3 — Per-op constraints (arity + required attributes)
// =============================================================================

void Validator::checkOpConstraints(
    const sir::Operation& op, ValidationReport& report) const
{
    const std::string mn(op.mnemonic());

    auto requireOperands = [&](size_t expected) {
        if (op.numOperands() != expected)
            report.diagnostics.push_back({
                ValidationError::Severity::Error, mn, "",
                mn + " requires exactly " + std::to_string(expected) +
                    " operand(s), got " + std::to_string(op.numOperands())});
    };

    auto requireAttr = [&](std::string_view key) {
        if (!op.hasAttribute(key))
            report.diagnostics.push_back({
                ValidationError::Severity::Error, mn, "",
                mn + " is missing required attribute '" + std::string(key) + "'"});
    };

    if (mn == "sc_high.matmul") {
        requireOperands(2);

    } else if (mn == "sc_high.gemm") {
        // Bias is optional — 2 or 3 operands both valid.
        if (op.numOperands() < 2 || op.numOperands() > 3)
            report.diagnostics.push_back({
                ValidationError::Severity::Error, mn, "",
                "Gemm requires 2 or 3 operands, got " +
                    std::to_string(op.numOperands())});

    } else if (mn == "sc_high.conv2d") {
        if (op.numOperands() < 2 || op.numOperands() > 3)
            report.diagnostics.push_back({
                ValidationError::Severity::Error, mn, "",
                "Conv2D requires 2 or 3 operands (bias optional), got " +
                    std::to_string(op.numOperands())});
        requireAttr("strides");
        requireAttr("pads");
        requireAttr("dilations");

    } else if (mn == "sc_high.batch_norm") {
        requireOperands(5);
        requireAttr("epsilon");

    } else if (mn == "sc_high.relu"  || mn == "sc_high.transpose" ||
               mn == "sc_high.reshape") {
        requireOperands(1);

    } else if (mn == "sc_high.add" || mn == "sc_high.sub" ||
               mn == "sc_high.mul" || mn == "sc_high.div") {
        requireOperands(2);

    } else if (mn == "sc_high.maxpool" || mn == "sc_high.avgpool") {
        requireOperands(1);
        requireAttr("kernel_shape");
        requireAttr("strides");

    } else if (mn == "sc_high.concat") {
        if (op.numOperands() < 2)
            report.diagnostics.push_back({
                ValidationError::Severity::Error, mn, "",
                "Concat requires at least 2 operands"});
        requireAttr("axis");

    } else if (mn == "sc_high.constant") {
        // Constants have no operands by definition.
        if (op.numOperands() != 0)
            report.diagnostics.push_back({
                ValidationError::Severity::Error, mn, "",
                "Constant op must have zero operands"});
        if (op.numResults() != 1)
            report.diagnostics.push_back({
                ValidationError::Severity::Error, mn, "",
                "Constant op must have exactly one result"});
    }
}

// =============================================================================
// Pass 4 — Type consistency
// =============================================================================

void Validator::checkTypeConsistency(
    const sir::Operation& op, ValidationReport& report) const
{
    // Ops where all operands must share the same DataType.
    static const std::unordered_set<std::string> kHomogeneousOps = {
        "sc_high.matmul", "sc_high.gemm",
        "sc_high.add",    "sc_high.sub",
        "sc_high.mul",    "sc_high.div",
        "sc_high.conv2d",
    };

    const std::string mn(op.mnemonic());
    if (kHomogeneousOps.find(mn) == kHomogeneousOps.end()) return;
    if (op.numOperands() < 2) return;

    const sir::DataType expected = op.operand(0)->dtype();
    for (size_t i = 1; i < op.numOperands(); ++i) {
        if (op.operand(i)->dtype() != expected) {
            report.diagnostics.push_back({
                ValidationError::Severity::Error,
                mn,
                std::string(op.operand(i)->id()),
                mn + ": operand " + std::to_string(i) +
                    " dtype '" +
                    std::string(sir::dtypeName(op.operand(i)->dtype())) +
                    "' does not match operand 0 dtype '" +
                    std::string(sir::dtypeName(expected)) + "'"});
        }
    }
}

// =============================================================================
// Pass 5 — Shape consistency
// =============================================================================

void Validator::checkShapeConsistency(
    const sir::Operation& op, ValidationReport& report) const
{
    const std::string mn(op.mnemonic());

    // Rule 0: Every result must have a resolved (non-placeholder) shape.
    for (size_t i = 0; i < op.numResults(); ++i) {
        const sir::Value* res = op.result(i);
        if (!res) continue;
        if (res->shape().dims.empty()) {
            report.diagnostics.push_back({
                ValidationError::Severity::Error,
                mn, std::string(res->id()),
                "Result '" + std::string(res->id()) +
                    "' still has an unresolved placeholder shape — "
                    "run ShapeInferencePass before Validator"});
        }
    }

    // Rule 1: Conv2D — input and filter must be rank-4 (NCHW).
    if (mn == "sc_high.conv2d" && op.numOperands() >= 2) {
        auto checkRank4 = [&](size_t idx, std::string_view label) {
            if (op.operand(idx)->shape().dims.size() != 4)
                report.diagnostics.push_back({
                    ValidationError::Severity::Error,
                    mn, std::string(op.operand(idx)->id()),
                    "Conv2D " + std::string(label) +
                        " must be rank-4 (NCHW)"});
        };
        checkRank4(0, "input");
        checkRank4(1, "filter");

        // Input channels must be consistent with filter: in[1] == filter[1]*group
        if (op.operand(0)->shape().dims.size() == 4 &&
            op.operand(1)->shape().dims.size() == 4)
        {
            int64_t group = op.getAttrAs<int64_t>("group").value_or(1);
            int64_t c_in     = op.operand(0)->shape().dims[1];
            int64_t c_filter = op.operand(1)->shape().dims[1];
            if (c_in  != sir::Shape::kDynamic &&
                c_filter != sir::Shape::kDynamic &&
                c_in != c_filter * group)
            {
                report.diagnostics.push_back({
                    ValidationError::Severity::Error, mn, "",
                    "Conv2D: input channels (" + std::to_string(c_in) +
                        ") != filter_channels (" + std::to_string(c_filter) +
                        ") * group (" + std::to_string(group) + ")"});
            }
        }
    }

    // Rule 2: MatMul — inner dimensions must contract.
    if (mn == "sc_high.matmul" && op.numOperands() == 2) {
        const auto& dA = op.operand(0)->shape().dims;
        const auto& dB = op.operand(1)->shape().dims;
        if (dA.size() >= 2 && dB.size() >= 2) {
            int64_t K_a = dA.back();
            int64_t K_b = dB[dB.size() - 2];
            if (K_a != sir::Shape::kDynamic &&
                K_b != sir::Shape::kDynamic &&
                K_a != K_b)
            {
                report.diagnostics.push_back({
                    ValidationError::Severity::Error, mn, "",
                    "MatMul: inner dimensions mismatch (" +
                        std::to_string(K_a) + " vs " +
                        std::to_string(K_b) + ")"});
            }
        }
    }

    // Rule 3: BatchNorm — scale and bias must be rank-1 with size == C.
    if (mn == "sc_high.batch_norm" && op.numOperands() == 5) {
        const auto& input_dims = op.operand(0)->shape().dims;
        if (input_dims.size() >= 2) {
            int64_t C = input_dims[1];  // NCHW channel dim
            for (size_t i : {size_t(1), size_t(2)}) {  // scale, bias
                const auto& d = op.operand(i)->shape().dims;
                if (d.size() != 1)
                    report.diagnostics.push_back({
                        ValidationError::Severity::Error,
                        mn, std::string(op.operand(i)->id()),
                        "BatchNorm operand " + std::to_string(i) +
                            " must be rank-1"});
                else if (d[0] != sir::Shape::kDynamic &&
                         C    != sir::Shape::kDynamic &&
                         d[0] != C)
                    report.diagnostics.push_back({
                        ValidationError::Severity::Error,
                        mn, std::string(op.operand(i)->id()),
                        "BatchNorm operand " + std::to_string(i) +
                            " size (" + std::to_string(d[0]) +
                            ") != input channel count (" +
                            std::to_string(C) + ")"});
            }
        }
    }
}

} // namespace seecpp::frontend