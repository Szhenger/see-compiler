#include "include/frontend/onnx_ingressor.hpp"
#include "include/frontend/shape_inference.hpp"
#include "include/frontend/validator.hpp"
#include "include/middle-end/pass_manager.hpp"
#include "include/utility/logger.hpp"
#include "include/utility/weight_buffer.hpp"

#include <iostream>
#include <string_view>

// =============================================================================
// Global weight buffer
// Defined here (translation-unit owner); declared `extern` in onnx_ingressor.cpp
// =============================================================================
namespace seecpp::utility {
    WeightBuffer global_weight_buffer;
}

// =============================================================================
// Helpers
// =============================================================================

namespace {

/// Print a ValidationReport to the logger.
/// Warnings are surfaced; errors are printed then the caller should exit.
void reportValidation(const seecpp::frontend::ValidationReport& report) {
    for (const auto& d : report.diagnostics) {
        const std::string prefix =
            d.op_mnemonic.empty() ? "" : "[" + d.op_mnemonic + "] ";
        if (d.isWarning())
            seecpp::utility::Logger::warn("Validator: " + prefix + d.message);
        else
            seecpp::utility::Logger::error("Validator: " + prefix + d.message);
    }
}

} // anonymous namespace

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {

    // --- Startup configuration ---
    // In release builds, silence INFO traces with setLevel(Warn).
    seecpp::utility::Logger::setLevel(seecpp::utility::LogLevel::Info);

    if (argc < 3) {
        std::cerr
            << "Usage: seecpp-frontend <input.onnx> <output.sir>\n"
            << "  input.onnx  — ONNX model to compile\n"
            << "  output.sir  — destination for serialised SIR\n";
        return 1;
    }

    const std::string_view input_path  = argv[1];
    const std::string_view output_path = argv[2];

    seecpp::utility::Logger::info("SeeC++ frontend pipeline starting");
    seecpp::utility::Logger::info(
        std::string("  input  : ") + std::string(input_path));
    seecpp::utility::Logger::info(
        std::string("  output : ") + std::string(output_path));

    // =========================================================================
    // Stage 1 — Ingestion
    // OnnxIngressor owns the Block it produces; we take ownership on success.
    // =========================================================================
    seecpp::frontend::OnnxIngressor ingressor;
    auto ingest_result = ingressor.ingest(input_path);

    if (!ingest_result) {
        const auto& err = ingest_result.error();
        seecpp::utility::Logger::error(
            "Ingestion failed" +
            (err.node_name.empty() ? "" : " at node '" + err.node_name + "'") +
            ": " + err.message);
        return 1;
    }

    std::unique_ptr<seecpp::sir::Block> model_block = std::move(*ingest_result);

    // =========================================================================
    // Stage 2 — Shape Inference
    // Resolves all Shape{} placeholders left by the ingressor.
    // =========================================================================
    seecpp::frontend::ShapeInferencePass shape_pass;
    auto shape_result = shape_pass.run(*model_block);

    if (!shape_result) {
        const auto& err = shape_result.error();
        seecpp::utility::Logger::error(
            "Shape inference failed" +
            (err.op_mnemonic.empty() ? "" : " at op '" + err.op_mnemonic + "'") +
            (err.value_id.empty()    ? "" : " value '" + err.value_id    + "'") +
            ": " + err.message);
        return 1;
    }

    // =========================================================================
    // Stage 3 — Validation
    // Multi-pass structural and semantic check on the fully-shaped IR.
    // All diagnostics are collected before we decide to abort.
    // =========================================================================
    seecpp::frontend::Validator validator;
    auto report = validator.validate(*model_block);

    reportValidation(report);

    if (report.hasErrors()) {
        seecpp::utility::Logger::error(
            "Validation failed with " +
            std::to_string(report.diagnostics.size()) +
            " issue(s) — aborting compilation");
        return 1;
    }

    // =========================================================================
    // Stage 4 — Middle-End Pass Pipeline
    // Passes are registered here; the PassManager drives execution, inter-pass
    // validation, and timing instrumentation.
    //
    // Typical pipeline order for a DL compiler:
    //   1. Lowering    : sc_high.* -> sc_low.*  (ConvLoweringPass, GemmLoweringPass)
    //   2. Fusion      : merge adjacent elementwise ops (OperatorFusionPass)
    //   3. Constant folding / DCE
    //   4. Buffer allocation : sc_low.* -> sc_mem.*
    // =========================================================================
    seecpp::middle::PassManager pm;

    // Register passes as they are implemented, e.g.:
    //   pm.addPass<seecpp::middle::ConvLoweringPass>()
    //     .addPass<seecpp::middle::OperatorFusionPass>()
    //     .addPass<seecpp::middle::DeadCodeEliminationPass>();

    auto pm_result = pm.run(*model_block);

    if (!pm_result) {
        const auto& err = pm_result.error();
        seecpp::utility::Logger::error(
            "Middle-end pass '" + err.pass_name + "' failed: " + err.message);

        // Surface any validation diagnostics the PassManager captured.
        reportValidation(pm.lastReport());
        return 1;
    }

    // =========================================================================
    // Stage 5 — Serialisation
    // Write the lowered SIR to disk for the Python codegen backend to consume.
    //
    // TODO: implement serializeBlockToFile(output_path, *model_block)
    //   Options: flatbuffers (zero-copy, Python-friendly), protobuf, or a
    //   custom text format printed via Block::print() for debugging.
    // =========================================================================
    seecpp::utility::Logger::info(
        "Serialising SIR to '" + std::string(output_path) + "'");

    // serializeBlockToFile(output_path, *model_block);

    seecpp::utility::Logger::info(
        "SeeC++ frontend pipeline completed — " +
        std::to_string(model_block->numOps()) + " op(s) in output block");

    return 0;
}