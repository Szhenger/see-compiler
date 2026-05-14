#pragma once

#include "middle-end/sir.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>
#include <expected>

namespace seecpp::frontend {

struct ValidationError {
    enum class Severity { Warning, Error };

    Severity severity = Severity::Error;
    std::string op_mnemonic;
    std::string value_id;
    std::string message;

    bool isError() const { return severity == Severity::Error; }
    bool isWarning() const { return severity == Severity::Warning; }
};

struct ValidationReport {
    std::vector<ValidationError> diagnostics;

    bool hasErrors() const {
        for (const auto& d : diagnostics)
            if (d.isError()) return true;
        return false;
    }
    bool empty() const { return diagnostics.empty(); }
};

class Validator {
public:
    Validator() = default;

    [[nodiscard]]
    ValidationReport validate(const sir::Block& block);

private:
    void checkSSALinks(
        const sir::Block& block, ValidationReport& report) const;

    void checkTopologicalOrder(
        const sir::Block& block, ValidationReport& report) const;

    void checkOpConstraints(
        const sir::Operation& op, ValidationReport& report) const;

    void checkTypeConsistency(
        const sir::Operation& op, ValidationReport& report) const;

    void checkShapeConsistency(
        const sir::Operation& op, ValidationReport& report) const;

    static const std::unordered_set<std::string> kSupportedOps;
};

} // namespace seecpp::frontend
