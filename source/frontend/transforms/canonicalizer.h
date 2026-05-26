#ifndef SEECPP_TRANSFORMS_CANONICALIZER_H_
#define SEECPP_TRANSFORMS_CANONICALIZER_H_

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "seecpp/sir/sir.h"

namespace seecpp::transforms {

/// @brief Safely mutates the IR during pattern matching. Acts as an IR builder
/// that tracks where new operations should be inserted.
class PatternRewriter {
 public:
  explicit PatternRewriter(sir::Block* block) : block_(block), ip_(nullptr) {}
  ~PatternRewriter() = default;

  /// @brief Sets the insertion point immediately before the given operation.
  void SetInsertionPoint(sir::Operation* op);

  /// @brief Creates a new operation at the current insertion point.
  sir::Operation* CreateOp(std::string_view mnemonic,
                           const std::vector<sir::Value*>& operands,
                           const std::vector<sir::DataType>& result_types,
                           const std::vector<sir::Shape>& result_shapes);

  /// @brief Replaces an operation and remaps its outputs to new values.
  void ReplaceOp(sir::Operation* old_op,
                 const std::vector<sir::Value*>& new_values);

  /// @brief Erases an operation from the block cleanly.
  void EraseOp(sir::Operation* op);

 private:
  sir::Block* block_;
  sir::Operation* ip_;  // Current insertion point
};

/// @brief Abstract base class for matching and rewriting graph operations.
class RewritePattern {
 public:
  explicit RewritePattern(std::string target_mnemonic)
      : target_mnemonic_(std::move(target_mnemonic)) {}
  virtual ~RewritePattern() = default;

  std::string_view GetTargetMnemonic() const { return target_mnemonic_; }

  /// @brief Analyzes the op and performs structural replacement if matched.
  /// @return True if the graph was mutated; false if no change occurred.
  virtual bool MatchAndRewrite(sir::Operation* op,
                               PatternRewriter& rewriter) const = 0;

 private:
  std::string target_mnemonic_;
};

/// @brief Fixed-point optimization driver that applies progressive patterns.
class Canonicalizer {
 public:
  Canonicalizer() = default;
  ~Canonicalizer() = default;

  Canonicalizer(const Canonicalizer&) = delete;
  Canonicalizer& operator=(const Canonicalizer&) = delete;

  /// @brief Enrolls an operator-specific decomposition rule into the pipeline.
  void AddPattern(std::unique_ptr<RewritePattern> pattern);

  /// @brief Iterates over the block applying patterns until convergence.
  /// @return True if any modifications were made to the graph structure.
  bool RunOnBlock(sir::Block* block);

 private:
  using PatternList = std::vector<std::unique_ptr<RewritePattern>>;
  
  std::unordered_map<std::string, PatternList> patterns_;
  
  // Safeguards against infinite loops if patterns exhibit cyclical dependencies
  static constexpr int kMaxIterations = 10;
};

}  // namespace seecpp::transforms

#endif  // SEECPP_TRANSFORMS_CANONICALIZER_H_
