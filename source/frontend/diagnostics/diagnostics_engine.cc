#include "source/frontend/diagnostics_engine.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace seecpp::frontend {

namespace {

// ANSI escape codes for rich terminal output
constexpr std::string_view kReset     = "\033[0m";
constexpr std::string_view kRed       = "\033[1;31m";
constexpr std::string_view kMagenta   = "\033[1;35m";
constexpr std::string_view kCyan      = "\033[1;36m";
constexpr std::string_view kWhiteBold = "\033[1;37m";

std::string LevelToString(DiagnosticLevel level) {
  switch (level) {
    case DiagnosticLevel::Note:    return "note: ";
    case DiagnosticLevel::Warning: return "warning: ";
    case DiagnosticLevel::Error:   return "error: ";
    case DiagnosticLevel::Fatal:   return "fatal error: ";
  }
  return "unknown: ";
}

std::string LevelToColor(DiagnosticLevel level) {
  switch (level) {
    case DiagnosticLevel::Note:    return std::string(kCyan);
    case DiagnosticLevel::Warning: return std::string(kMagenta);
    case DiagnosticLevel::Error:
    case DiagnosticLevel::Fatal:   return std::string(kRed);
  }
  return "";
}

}  // namespace

void DiagnosticsEngine::Report(DiagnosticLevel level, const SourceLocation& loc,
                               std::string_view message) {
  if (level == DiagnosticLevel::Error || level == DiagnosticLevel::Fatal) {
    ++error_count_;
  }

  // 1. Print Header: file:line:col: severity: message
  std::cerr << kWhiteBold << loc.file_path << ":" << loc.line << ":"
            << loc.column << ": " << kReset;
  
  std::cerr << LevelToColor(level) << LevelToString(level) << kReset
            << kWhiteBold << message << kReset << "\n";

  // 2. Fetch and print the specific line of code
  std::string source_line = GetSourceLine(loc.file_path, loc.line);
  if (!source_line.empty()) {
    PrintSourceContext(loc, source_line);
  }

  // 3. Terminate immediately if the error is unrecoverable
  if (level == DiagnosticLevel::Fatal) {
    std::exit(EXIT_FAILURE);
  }
}

std::string DiagnosticsEngine::GetSourceLine(const std::string& file_path,
                                             size_t line) const {
  // For production, this I/O should be cached using a dedicated 
  // SourceManager that memory-maps files, but standard I/O works for the MVP.
  std::ifstream file(file_path);
  if (!file.is_open()) return "";

  std::string current_line;
  size_t current_line_num = 1;
  while (std::getline(file, current_line)) {
    if (current_line_num == line) {
      return current_line;
    }
    ++current_line_num;
  }
  return "";
}

void DiagnosticsEngine::PrintSourceContext(
    const SourceLocation& loc, const std::string& source_line) const {
  
  std::string line_num_str = std::to_string(loc.line);
  std::string padding(line_num_str.length(), ' ');

  // Print the line of code with the line number margin
  // e.g., " 10 |     int x = foo;"
  std::cerr << " " << line_num_str << " | " << source_line << "\n";
  
  // Print the margin for the squiggles
  // e.g., "    | "
  std::cerr << " " << padding << " | ";
  
  // Calculate spaces up to the caret (1-based column index to 0-based)
  size_t spaces_to_caret = (loc.column > 0) ? loc.column - 1 : 0;
  for (size_t i = 0; i < spaces_to_caret; ++i) {
    // Preserve tabs from the source to keep alignment strictly accurate
    if (i < source_line.length() && source_line[i] == '\t') {
      std::cerr << '\t';
    } else {
      std::cerr << ' ';
    }
  }

  // Draw the caret and squiggles based on token length
  std::cerr << kRed << "^";
  for (size_t i = 1; i < loc.length; ++i) {
    std::cerr << "~";
  }
  std::cerr << kReset << "\n";
}

}  // namespace seecpp::frontend
