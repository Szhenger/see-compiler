#include "include/frontend/onnx_ingressor.hpp"
#include "include/frontend/shape_inference.hpp"
#include "include/frontend/validator.hpp"
#include "include/middle-end/pass_manager.hpp"
#include "include/utility/logger.hpp"
#include "include/utility/weight_buffer.hpp"

#include <iostream>
#include <string_view>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;
using seecpp::utility::Logger;

namespace seecpp::utility {
    // Explicit ownership of the weight buffer for the lifetime of the process.
    WeightBuffer global_weight_buffer;
}

namespace seecpp::frontend {

class FrontendDriver {
public:
    struct Config {
        fs::path input_path;
        fs::path output_path;
        bool verbose = false;
    };

    explicit FrontendDriver(Config cfg) : config_(std::move(cfg)) {}

    int run() {
        setupLogger();
        Logger::info("SeeC++ frontend pipeline starting");

        // 1. Ingestion
        auto model_block = ingest();
        if (!model_block) return 1;

        // 2. Shape Inference
        if (!runShapeInference(*model_block)) return 1;

        // 3. Structural Validation
        if (!validate(*model_block)) return 1;

        // 4. Optimization & Lowering
        if (!runMiddleEnd(*model_block)) return 1;

        // 5. Serialization
        return serialize(*model_block) ? 0 : 1;
    }

private:
    Config config_;

    void setupLogger() {
        Logger::setLevel(config_.verbose ? utility::LogLevel::Debug : utility::LogLevel::Info);
    }

    void reportDiagnostics(const ValidationReport& report) {
        for (const auto& d : report.diagnostics) {
            std::string msg = d.op_mnemonic.empty() ? "" : "[" + d.op_mnemonic + "] ";
            msg += d.message;
            
            if (d.isWarning()) Logger::warn("Validator: " + msg);
            else Logger::error("Validator: " + msg);
        }
    }

    std::unique_ptr<sir::Block> ingest() {
        OnnxIngressor ingressor;
        auto result = ingressor.ingest(config_.input_path.string());
        
        if (!result) {
            const auto& err = result.error();
            Logger::error("Ingestion failed" + 
                (err.node_name.empty() ? "" : " at node '" + err.node_name + "'") + 
                ": " + err.message);
            return nullptr;
        }
        return std::move(*result);
    }

    bool runShapeInference(sir::Block& block) {
        ShapeInferencePass shape_pass;
        auto result = shape_pass.run(block);
        
        if (!result) {
            const auto& err = result.error();
            Logger::error("Shape inference failed at op '" + err.op_mnemonic + "': " + err.message);
            return false;
        }
        return true;
    }

    bool validate(sir::Block& block) {
        Validator validator;
        auto report = validator.validate(block);
        reportDiagnostics(report);
        
        if (report.hasErrors()) {
            Logger::error("Validation failed — aborting compilation");
            return false;
        }
        return true;
    }

    bool runMiddleEnd(sir::Block& block) {
        middle::PassManager pm;
        // Pipeline configuration would typically be loaded from a config file or CLI flags
        auto result = pm.run(block);
        
        if (!result) {
            const auto& err = result.error();
            Logger::error("Middle-end pass '" + err.pass_name + "' failed: " + err.message);
            reportDiagnostics(pm.lastReport());
            return false;
        }
        return true;
    }

    bool serialize(const sir::Block& block) {
        Logger::info("Serialising SIR to '" + config_.output_path.string() + "'");
        // Placeholder for future Flatbuffers/Protobuf implementation
        Logger::info("SeeC++ frontend pipeline completed — " + 
                     std::to_string(block.numOps()) + " op(s) generated");
        return true;
    }
};

} // namespace seecpp::frontend

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: seecpp-frontend <input.onnx> <output.sir> [--verbose]\n";
        return 1;
    }

    seecpp::frontend::FrontendDriver::Config cfg;
    cfg.input_path = argv[1];
    cfg.output_path = argv[2];
    
    for (int i = 3; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--verbose") cfg.verbose = true;
    }

    seecpp::frontend::FrontendDriver driver(cfg);
    return driver.run();
}
