#pragma once

#include "middle-end/sir.hpp"
#include "frontend/validator.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <expected>
#include <chrono>

namespace seecpp::middle {

struct PassError {
    std::string pass_name;
    std::string message;
    // Possible expansion: int error_code;
};

using PassResult = std::expected<void, PassError>;

class IPass {
public:
    virtual ~IPass() = default;
    virtual std::string_view name() const = 0;
    
    [[nodiscard]]
    virtual PassResult runOnBlock(sir::Block& block) = 0;

    virtual bool requiresValidation() const { return true; }
};

class PassManager {
public:
    struct PassTiming {
        std::string pass_name;
        double elapsed_ms = 0.0;
    };

    PassManager() = default;
    ~PassManager() = default;

    PassManager(const PassManager&) = delete;
    PassManager& operator=(const PassManager&) = delete;
    PassManager(PassManager&&) = default;
    PassManager& operator=(PassManager&&) = default;

    template <typename PassT, typename... Args>
    PassManager& addPass(Args&&... args) {
        pipeline_.push_back(std::make_unique<PassT>(std::forward<Args>(args)...));
        return *this;
    }

    PassManager& addPass(std::unique_ptr<IPass> pass) {
        if (pass) pipeline_.push_back(std::move(pass));
        return *this;
    }

    [[nodiscard]]
    PassResult run(sir::Block& block);

    const std::vector<PassTiming>& timings() const noexcept { return timings_; }
    const frontend::ValidationReport& lastReport() const noexcept { return last_report_; }
    
    void printTimings() const;
    void clear() { pipeline_.clear(); timings_.clear(); }

private:
    std::vector<std::unique_ptr<IPass>> pipeline_;
    std::vector<PassTiming> timings_;
    frontend::ValidationReport last_report_;
    frontend::Validator validator_;
};

} // namespace seecpp::middle
