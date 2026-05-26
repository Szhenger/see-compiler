#include "source/frontend/frontend_driver.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "include/utility/logger.hpp"
#include "include/utility/weight_buffer.hpp"
#include "seecpp/sir/sir.h"
#include "source/frontend/diagnostics_engine.h"
#include "source/frontend/onnx_ingressor.h"
#include "source/frontend/shape_inference.h"
#include "source/frontend/validator.h"
#include "source/middle_end/pass_manager.h"

namespace fs = std::filesystem;

namespace seecpp::utility {
// Explicit ownership of the weight buffer for the lifetime of the process.
WeightBuffer global_weight_buffer;
}  // namespace seecpp::utility

namespace seecpp::frontend {

FrontendDriver::FrontendDriver(Config config)
    : config_(std::move(config)),
      diag_engine_(std::make_unique<DiagnosticsEngine>()) {}

int FrontendDriver::Run() {
  SetupLogger();
  utility::Logger::info("SeeC++ frontend pipeline starting");

  // 1. Ingestion
  auto model_block = Ingest();
  if (!model_block) return 1;

  // 2. Shape Inference
  if (!RunShapeInference(*model_block)) return 1;

  // 3. Structural Validation
  if (!Validate(*model_block)) return 1;

  // 4. Optimization & Lowering
  if (!RunMiddleEnd(*model_block)) return 1;

  // 5. Serialization
  return Serialize(*model_block) ? 0 : 1;
}

void FrontendDriver::SetupLogger() const {
  utility::Logger::setLevel(config_.verbose ? utility::LogLevel::Debug
                                            : utility::LogLevel::Info);
}

void FrontendDriver::ReportDiagnostics(const ValidationReport& report) {
  // Map validation errors into the DiagnosticsEngine.
  // Note: ONNX nodes don't inherently have source line numbers, so we 
  // initialize a default SourceLocation tied to the input file.
  SourceLocation loc;
  loc.file_path = config_.input_path.string();
  loc.line = 0;
  loc.column = 0;
  loc.length = 1;

  for (const auto& diag : report.diagnostics) {
    DiagnosticLevel level = 
        (diag.severity == ValidationError::Severity::Warning)
            ? DiagnosticLevel::Warning
            : DiagnosticLevel::Error;

    std::string msg = diag.op_mnemonic.empty() ? "" : "[" + diag.op_mnemonic + "] ";
    msg += diag.message;

    diag_engine_->Report(level, loc, msg);
  }
}

std::unique_ptr<sir::Block> FrontendDriver::Ingest() {
  OnnxIngressor ingressor;
  auto result = ingressor.ingest(config_.input_path.string());

  if (!result) {
    const auto& err = result.error();
    std::string node_info =
        err.node_name.empty() ? "" : " at node '" + err.node_name + "'";
    utility::Logger::error("Ingestion failed" + node_info + ": " + err.message);
    return nullptr;
  }
  return std::move(*result);
}

bool FrontendDriver::RunShapeInference(sir::Block& block) {
  ShapeInferencePass shape_pass;
  auto result = shape_pass.run(block);

  if (!result) {
    const auto& err = result.error();
    utility::Logger::error("Shape inference failed at op '" + err.op_mnemonic +
                           "': " + err.message);
    return false;
  }
  return true;
}

bool FrontendDriver::Validate(sir::Block& block) {
  Validator validator;
  ValidationReport report = validator.Validate(block);
  
  if (report.HasErrors() || !report.diagnostics.empty()) {
    ReportDiagnostics(report);
  }

  if (report.HasErrors()) {
    utility::Logger::error("Validation failed — aborting compilation");
    return false;
  }
  return true;
}

bool FrontendDriver::RunMiddleEnd(sir::Block& block) {
  middle::PassManager pm;
  // Pipeline configuration would typically be loaded from a config or CLI flags
  auto result = pm.run(block);

  if (!result) {
    const auto& err = result.error();
    utility::Logger::error("Middle-end pass '" + err.pass_name +
                           "' failed: " + err.message);
    
    // Assuming the PassManager also returns a ValidationReport-compatible struct
    // ReportDiagnostics(pm.lastReport()); 
    return false;
  }
  return true;
}

bool FrontendDriver::Serialize(const sir::Block& block) const {
  utility::Logger::info("Serializing SIR to '" + config_.output_path.string() +
                        "'");
  // Placeholder for future Flatbuffers/Protobuf serialization
  utility::Logger::info("SeeC++ frontend pipeline completed — " +
                        std::to_string(block.operations().size()) +
                        " op(s) generated");
  return true;
}

}  // namespace seecpp::frontend
