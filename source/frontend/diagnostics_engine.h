#ifndef SEECPP_FRONTEND_DIAGNOSTICS_ENGINE_H_
#define SEECPP_FRONTEND_DIAGNOSTICS_ENGINE_H_

#include <string>
#include <string_view>

namespace seecpp::frontend {

/// @brief Represents the severity of a compiler diagnostic message.
enum class DiagnosticLevel {
  Note,
  Warning,
  Error,
  Fatal
};

/// @brief Pinpoints the exact location and size of a token in the source code.
struct SourceLocation {
  std::string file_path;
  size_t line = 0;
  size_t column = 0;
  size_t length = 1;  // Controls the length of the '~~~' squiggles
};

/// @brief A Clang-style terminal diagnostics printer. Handles ANSI coloring,
/// source line retrieval, and caret/squiggle formatting for precise errors.
class DiagnosticsEngine {
 public:
  DiagnosticsEngine() = default;
  ~DiagnosticsEngine() = default;

  DiagnosticsEngine(const DiagnosticsEngine&) = delete;
  DiagnosticsEngine& operator=(const DiagnosticsEngine&) = delete;

  /// @brief Reports a message to the terminal with rich source context.
  void Report(DiagnosticLevel level, const SourceLocation& loc,
              std::string_view message);

  /// @brief Returns the total number of errors and fatal errors reported.
  [[nodiscard]] size_t GetErrorCount() const { return error_count_; }

 private:
  size_t error_count_ = 0;

  /// @brief Fetches a specific line of text directly from the source file.
  std::string GetSourceLine(const std::string& file_path, size_t line) const;

  /// @brief Prints the Clang-style source context block (line numbers, carets).
  void PrintSourceContext(const SourceLocation& loc, 
                          const std::string& source_line) const;
};

}  // namespace seecpp::frontend

#endif  // SEECPP_FRONTEND_DIAGNOSTICS_ENGINE_H_
