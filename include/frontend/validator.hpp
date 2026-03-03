#pragma once

#include "middle-end/sir.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>
#include <expected>

namespace seecpp::frontend {

// =============================================================================
// Diagnostics
// =============================================================================

struct ValidationError {
    enum class Severity { Warning, Error };

    Severity    severity   = Severity::Error;
    std::string op_mnemonic;   // mnemonic of the offending op (empty = block-level)
    std::string value_id;      // offending Value id, if applicable
    std::string message;

    bool isError()   const { return severity == Severity::Error;   }
    bool isWarning() const { return severity == Severity::Warning; }
};

// Convenience: a full validation report (warnings + errors together).
struct ValidationReport {
    std::vector<ValidationError> diagnostics;

    bool hasErrors() const {
        for (const auto& d : diagnostics)
            if (d.isError()) return true;
        return false;
    }
    bool empty() const { return diagnostics.empty(); }
};

// =============================================================================
// Validator
// =============================================================================

/// Multi-pass structural and semantic validator for a fully-ingested,
/// shape-inferred SIR Block.
///
/// Pass order:
///   1. checkSSALinks          — no null pointers, no undefined values
///   2. checkTopologicalOrder  — use before def, acyclic DAG
///   3. checkOpConstraints     — per-op arity, attribute presence
///   4. checkTypeConsistency   — no cross-dtype operand mismatches
///   5. checkShapeConsistency  — operand shapes satisfy op contracts
///
/// All passes run even if an earlier one finds errors, so you receive the
/// full diagnostic set in one shot rather than fix-one-rerun cycles.
class Validator {
public:
    Validator() = default;

    /// Run all validation passes over `block`.
    /// Returns a ValidationReport; check report.hasErrors() for hard failures.
    [[nodiscard]]
    ValidationReport validate(const sir::Block& block);

private:
    // --- Pass 1: SSA structural integrity ---
    // Ensures every operand pointer is non-null and points to a Value that
    // is reachable from this block (no dangling cross-block references).
    void checkSSALinks(
        const sir::Block& block, ValidationReport& report) const;

    // --- Pass 2: Topological order (DAG check) ---
    // Verifies that every Value is defined before it is used within the block.
    // Detects cycles that would prevent linear scheduling.
    void checkTopologicalOrder(
        const sir::Block& block, ValidationReport& report) const;

    // --- Pass 3: Per-op constraints (arity + required attributes) ---
    // Each op in supported_ops has a known operand count and required
    // attribute set. Violations are hard errors.
    void checkOpConstraints(
        const sir::Operation& op, ValidationReport& report) const;

    // --- Pass 4: Type consistency ---
    // For ops with multiple operands (MatMul, Add, Conv2D) all operands
    // must share the same DataType unless an explicit cast op is present.
    void checkTypeConsistency(
        const sir::Operation& op, ValidationReport& report) const;

    // --- Pass 5: Shape consistency ---
    // Verifies operand-result shape contracts specific to each op:
    //   - All results must have a non-empty shape (no Shape{} placeholders)
    //   - Conv2D: input rank-4, filter rank-4, channel dims compatible
    //   - BatchNorm: scale/bias rank-1, size == input channel count
    //   - MatMul: inner dimensions must contract correctly
    void checkShapeConsistency(
        const sir::Operation& op, ValidationReport& report) const;

    // --- Allowlist ---
    // Only ops in this set are permitted to reach the middle-end lowering.
    // sc_high.unknown is intentionally excluded — unknowns must be resolved
    // or explicitly whitelisted before the block is considered valid.
    static const std::unordered_set<std::string> kSupportedOps;
};

} // namespace seecpp::frontend