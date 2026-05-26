#ifndef SEECPP_FRONTEND_FRONTEND_DRIVER_H_
#define SEECPP_FRONTEND_FRONTEND_DRIVER_H_

#include <filesystem>
#include <memory>

#include "seecpp/sir/sir.h"
#include "source/frontend/diagnostics_engine.h"
#include "source/frontend/validator.h"

namespace seecpp::frontend {

/// @brief Orchestrates the frontend compilation pipeline: Ingestion, Shape
/// Inference, Validation, Optimization, and Serialization.
class FrontendDriver {
 public:
  struct Config {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    bool verbose = false;
  };

  explicit FrontendDriver(Config config);
  ~FrontendDriver() = default;

  FrontendDriver(const FrontendDriver&) = delete;
  FrontendDriver& operator=(const FrontendDriver&) = delete;

  /// @brief Executes the full frontend pipeline.
  /// @return 0 on success, non-zero on failure.
  int Run();

 private:
  Config config_;
  std::unique_ptr<DiagnosticsEngine> diag_engine_;

  void SetupLogger() const;
  void ReportDiagnostics(const ValidationReport& report);

  std::unique_ptr<sir::Block> Ingest();
  bool RunShapeInference(sir::Block& block);
  bool Validate(sir::Block& block);
  bool RunMiddleEnd(sir::Block& block);
  bool Serialize(const sir::Block& block) const;
};

}  // namespace seecpp::frontend

#endif  // SEECPP_FRONTEND_FRONTEND_DRIVER_H_
