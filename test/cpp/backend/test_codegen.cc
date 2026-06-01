// test/cpp/backend/test_codegen.cc
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include "include/backend/codegen_driver.h"
#include "seecpp/sir/sir.h"
#include "seecpp/utility/weight_buffer.h"

namespace seecpp::backend::testing {

class CodegenDriverTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "seecpp_codegen_tests";
        std::filesystem::create_directories(test_dir_);
        valid_output_bin_ = test_dir_ / "compiled_topology.see";
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    // Helper to generate a structurally sound block for baseline testing
    sir::Block CreateValidGraph() {
        sir::Block block;
        // In reality, this would populate basic nodes like Add or MatMul
        block.add_tensor("Input", {1, 128});
        return block;
    }

    std::filesystem::path test_dir_;
    std::filesystem::path valid_output_bin_;
};

// --- Test Cases ---

TEST_F(CodegenDriverTest, ReturnsErrorWhenInstructionSelectionFails) {
    CodegenDriver driver;
    sir::Block bad_block = CreateValidGraph();
    
    // Inject a dummy operation that the InstructionSelector cannot lower
    bad_block.add_unsupported_hardware_op("UNMAPPABLE_OP"); 
    
    utility::WeightBuffer weights; // Empty weights for now

    auto result = driver.Run(bad_block, weights, valid_output_bin_.string());

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().phase, "instruction_selection");
}

TEST_F(CodegenDriverTest, ReturnsErrorWhenWeightPackingFails) {
    CodegenDriver driver;
    sir::Block valid_block = CreateValidGraph();
    
    // Add an operation requiring parameters (e.g., MatMul expecting 4096 bytes)
    valid_block.add_parameterized_op("MatMul_1", /*required_bytes=*/4096);
    
    // Provide a mismatched, insufficient weight buffer
    utility::WeightBuffer insufficient_weights(1024); 

    auto result = driver.Run(valid_block, insufficient_weights, valid_output_bin_.string());

    // Fails specifically at Phase 3 because the graph expects weights that don't exist
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().phase, "weight_packing");
}

TEST_F(CodegenDriverTest, ReturnsErrorWhenSerializationFails) {
    CodegenDriver driver;
    sir::Block valid_block = CreateValidGraph();
    utility::WeightBuffer valid_weights(4096); // Assuming matching weights

    // Force an I/O failure by specifying an illegal output target
    // (e.g., a directory path instead of a file, or a root system path)
    std::string illegal_output_path = "/sys/class/read_only_test.see";

    auto result = driver.Run(valid_block, valid_weights, illegal_output_path);

    // Passes Phases 1-3, but fails to write the physical file
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().phase, "serialization");
}

TEST_F(CodegenDriverTest, HappyPathProducesValidBinaryFile) {
    CodegenDriver driver;
    sir::Block valid_block = CreateValidGraph();
    utility::WeightBuffer valid_weights(4096); 

    // Assuming CreateValidGraph() and valid_weights align perfectly
    auto result = driver.Run(valid_block, valid_weights, valid_output_bin_.string());

    // 1. Check compiler execution success
    ASSERT_TRUE(result.has_value()) 
        << "Compilation failed during phase: " << result.error().phase 
        << " - " << result.error().message;

    // 2. Check filesystem state (Artifact Verification)
    ASSERT_TRUE(std::filesystem::exists(valid_output_bin_));
    
    // 3. Ensure the file isn't empty (basic sanity check on the Serializer)
    EXPECT_GT(std::filesystem::file_size(valid_output_bin_), 0);
}

} // namespace seecpp::backend::testing
