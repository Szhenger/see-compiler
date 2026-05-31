#include "seecpp/backend/codegen_driver.hpp"
#include "seecpp/runtime/runtime_engine.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <filesystem>

using namespace seecpp;

int main() {
    std::cout << "======================================================\n";
    std::cout << "[SeeC++] End-to-End AOT Compiler Integration Test\n";
    std::cout << "======================================================\n\n";

    // =========================================================================
    // PHASE 1: The Compiler (Ahead-of-Time Generation)
    // =========================================================================
    std::cout << ">>> Phase 1: Compilation\n";

    // 1. Build the Intermediate Representation (SIR)
    // Note: This uses a mock representation of your frontend builder API.
    // In production, this block is populated by your graph parser (e.g., ONNX loader).
    sir::Block block;
    // sir::Value* input = block.addArgument({1, 4}, sir::DType::Float32);
    // sir::Value* weights = block.addConstant({4, 4}, ...);
    // sir::Value* matmul_out = block.addOperation("sc_low.matmul", {input, weights});
    // sir::Value* relu_out = block.addOperation("sc_low.relu", {matmul_out});

    // 2. Configure compilation target
    backend::CodegenOptions options;
    options.output_dir = "./build/artifacts";
    // Target flags would dictate whether to emit AVX-512 or NEON opcodes
    options.emit_multithreaded = true; 

    // 3. Run the compiler orchestration pipeline
    std::cout << " -> Running OffsetBinder and compiling SIR graph...\n";
    
    // Assuming CodegenDriver exposes a static Compile entrypoint
    auto codegen_result = backend::CodegenDriver::Compile(block, options);
    if (!codegen_result) {
        std::cerr << "[ERROR] Compilation failed: " << codegen_result.error().message << "\n";
        return 1;
    }

    std::string binary_path = options.output_dir + "/model.see";
    std::cout << " -> Successfully generated zero-deserialization binary: " << binary_path << "\n\n";


    // =========================================================================
    // PHASE 2: The Runtime (Edge / Bare-Metal Execution)
    // =========================================================================
    std::cout << ">>> Phase 2: VM Execution\n";

    // 1. Boot the lightweight virtual machine
    runtime::RuntimeEngine engine;
    
    // 2. Memory-map the binary directly from disk
    std::cout << " -> Mmapping .see file and allocating 64-byte padded arena...\n";
    auto load_result = engine.Load(binary_path);
    if (!load_result) {
        std::cerr << "[ERROR] Runtime load failed: " << load_result.error().message << "\n";
        return 1;
    }

    // 3. Prepare Dummy Input Data 
    // Example: A 1x4 vector with negative values to test the ReLU kernel
    std::vector<float> input_data = {-1.5f, 3.0f, -0.2f, 8.4f};
    std::cout << " -> Injecting input tensors...\n";
    auto set_result = engine.SetInput(input_data.data(), input_data.size());
    if (!set_result) {
        std::cerr << "[ERROR] Input injection failed: " << set_result.error().message << "\n";
        return 1;
    }

    // 4. Fire the Execution Loop!
    std::cout << " -> Invoking hardware kernels (AVX-512 / NEON)...\n";
    auto invoke_result = engine.Invoke();
    if (!invoke_result) {
        std::cerr << "[ERROR] Execution failed: " << invoke_result.error().message << "\n";
        return 1;
    }

    // 5. Retrieve and Validate Outputs
    // Based on our OffsetBinder, the first input tensor takes up a 64-byte padded block.
    // The next tensor (the output of the MatMul/ReLU) will begin exactly at offset 64.
    size_t expected_output_offset = 64; 
    const float* output_ptr = engine.GetOutput(expected_output_offset);
    
    if (!output_ptr) {
        std::cerr << "[ERROR] Failed to read memory arena at offset " << expected_output_offset << "\n";
        return 1;
    }

    std::cout << "\n[SUCCESS] Final Evaluated Output:\n";
    for (int i = 0; i < 4; ++i) {
        // If ReLU worked correctly, any negative inputs should now be clamped to 0.0
        std::cout << "    out[" << i << "] = " << output_ptr[i] << "\n";
    }

    std::cout << "\n======================================================\n";
    std::cout << "System Verified. Ready to initialize code freeze.\n";
    std::cout << "======================================================\n";
    return 0;
}
