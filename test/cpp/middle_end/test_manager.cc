// test/cpp/middle_end/test_manager.cc
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string_view>

#include "seecpp/middle_end/pass_manager.h"
#include "seecpp/middle_end/pass_context.h"
#include "seecpp/middle_end/passes/pass.h"
#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::testing {

using ::testing::_;
using ::testing::Return;
using ::testing::StrictMock;

// 1. Define a strict GMock object for the Pass interface.
// StrictMock ensures tests fail if any unexpected method is called.
class MockPass : public Pass {
 public:
  MOCK_METHOD(std::string_view, name, (), (const, override));
  MOCK_METHOD(bool, Run, (sir::Block&), (override));
};

// 2. Define a minimal Mock for the SIR Block if Verify() is virtual.
// If sir::Block is concrete, you can use a real instance and toggle a 
// "valid" state flag to simulate structural verification failures.
class MockBlock : public sir::Block {
 public:
  MOCK_METHOD(bool, Verify, (diagnostics::Engine*), (const, override));
};

class PassManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Default context: no dumping, no verification unless explicitly overridden
    context_.print_ir_after_all = false;
    context_.verify_each = false;
    context_.diags = nullptr; 
  }

  PassContext context_;
  MockBlock block_;
};

// --- Test Cases ---

TEST_F(PassManagerTest, IgnoresNullptrsGracefully) {
  PassManager pm(context_);
  
  pm.AddPass(nullptr);
  auto result = pm.Run(block_);

  // Pipeline should succeed, and return false (no mutations occurred)
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result.value());
}

TEST_F(PassManagerTest, AccumulatesMutationStateCorrectly) {
  PassManager pm(context_);

  auto pass1 = std::make_unique<StrictMock<MockPass>>();
  auto pass2 = std::make_unique<StrictMock<MockPass>>();

  // Keep raw pointers to configure GMock expectations after moving ownership
  auto* p1_ptr = pass1.get();
  auto* p2_ptr = pass2.get();

  EXPECT_CALL(*p1_ptr, name()).WillRepeatedly(Return("Pass1_NoMutate"));
  EXPECT_CALL(*p1_ptr, Run(_)).WillOnce(Return(false));

  EXPECT_CALL(*p2_ptr, name()).WillRepeatedly(Return("Pass2_Mutate"));
  EXPECT_CALL(*p2_ptr, Run(_)).WillOnce(Return(true));

  pm.AddPass(std::move(pass1));
  pm.AddPass(std::move(pass2));

  auto result = pm.Run(block_);

  // Since Pass2 mutated the graph, the overall result must be true
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result.value());
}

TEST_F(PassManagerTest, OptimizationBypassesVerificationOnZeroMutation) {
  // Force the context to verify each pass
  context_.verify_each = true;
  PassManager pm(context_);

  auto pass = std::make_unique<StrictMock<MockPass>>();
  EXPECT_CALL(*pass, name()).WillRepeatedly(Return("ZeroMutationPass"));
  
  // Pass does NOT mutate the graph
  EXPECT_CALL(*pass, Run(_)).WillOnce(Return(false));

  // CRITICAL ASSERTION: block_.Verify() must NEVER be called, 
  // respecting the optimization bypass in pass_manager.cc.
  EXPECT_CALL(block_, Verify(_)).Times(0);

  pm.AddPass(std::move(pass));
  auto result = pm.Run(block_);

  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result.value());
}

TEST_F(PassManagerTest, AbortsPipelineImmediatelyOnVerificationFailure) {
  context_.verify_each = true;
  PassManager pm(context_);

  auto pass_bad = std::make_unique<StrictMock<MockPass>>();
  auto pass_skipped = std::make_unique<StrictMock<MockPass>>();

  auto* bad_ptr = pass_bad.get();
  auto* skip_ptr = pass_skipped.get();

  EXPECT_CALL(*bad_ptr, name()).WillRepeatedly(Return("CorruptingPass"));
  EXPECT_CALL(*bad_ptr, Run(_)).WillOnce(Return(true)); // Mutates graph

  // Simulate the Block failing structural verification after mutation
  EXPECT_CALL(block_, Verify(_)).WillOnce(Return(false));

  // The second pass must NEVER run because the pipeline should short-circuit
  EXPECT_CALL(*skip_ptr, name()).Times(0);
  EXPECT_CALL(*skip_ptr, Run(_)).Times(0);

  pm.AddPass(std::move(pass_bad));
  pm.AddPass(std::move(pass_skipped));

  auto result = pm.Run(block_);

  // The Result must contain the expected error enum
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), PassError::kVerificationFailed);
}

}  // namespace seecpp::middle_end::testing
