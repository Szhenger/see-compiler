#include <iostream>
#include <string>
#include <memory>

#include "frontend/onnx_ingressor.hpp"
#include "frontend/shape_inference.hpp"
#include "frontend/validator.hpp"
#include "include/middle-end/pass_manager.hpp"
#include "include/utility/logger.hpp"
#include "include/utility/weight_buffer.hpp"

// Global weight buffer instance (referenced as 'extern' in the Ingressor)
namespace seecpp::utility {
    WeightBuffer global_weight_buffer;
}

using namespace seecpp;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: seecpp-frontend <input.onnx> <output.sir>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    utility::Logger::info("--- SeeC++ Frontend Pipeline Started ---");

    // 1. Initialize SIR Block
    auto model_block = std::make_unique<sir::Block>();

    // 2. Step 1: Ingestion
    frontend::OnnxIngressor ingressor;
    if (!ingressor.ingest(input_path, *model_block)) {
        utility::Logger::error("Ingestion stage failed.");
        return 1;
    }

    // 3. Step 2: Shape Inference (The MLIR 'Analysis' phase)
    frontend::ShapeInferenceEngine shape_engine;
    if (!shape_engine.infer(*model_block)) {
        utility::Logger::error("Shape inference stage failed.");
        return 1;
    }

    // 4. Step 3: Validation (The MLIR 'Verifer' phase)
    frontend::Validator validator;
    if (!validator.validate(*model_block)) {
        utility::Logger::error("Graph validation stage failed.");
        return 1;
    }

    // 5. Middle-End: Pass Manager (Lowering/Transformations)
    // Following MLIR rigor, we treat transformations as a sequence of passes
    middle_end::PassManager pm;
    
    // Example: pm.addPass(std::make_unique<middle_end::AutodiffPass>());
    // Example: pm.addPass(std::make_unique<middle_end::ConstantFolderPass>());
    
    if (!pm.run(*model_block)) {
        utility::Logger::error("Middle-end pass execution failed.");
        return 1;
    }

    // 6. Serialization (Save to .sir proto)
    // This is where we create a file that Python can easily parse
    utility::Logger::info("Serializing SIR to " + output_path);
    // [Implementation of serializeBlockToFile(output_path, *model_block)]

    utility::Logger::info("--- Frontend Compilation Successful ---");
    return 0;
}