#ifndef SEECPP_MIDDLE_END_MEMORY_ARENA_MAPPER_H_
#define SEECPP_MIDDLE_END_MEMORY_ARENA_MAPPER_H_

#include <cstdint>
#include <expected>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "seecpp/diagnostics/diagnostics_engine.h"
#include "seecpp/sir/sir.h"

namespace seecpp::middle_end::memory {

/// @brief Represents the physical location of a tensor within the arena.
struct TensorAllocation {
  size_t offset_bytes;
  size_t size_bytes;
};

/// @brief The final mapped blueprint for the workspace memory.
struct ArenaLayout {
  size_t total_arena_size_bytes = 0;
  std::unordered_map<const sir::Value*, TensorAllocation> mappings;
};

enum class MapperError {
  kDynamicShapeNotSupported,
  kInvalidTopology,
  kInternalAllocationError
};

/// @brief Computes non-overlapping memory offsets for transient tensors 
/// using linear-scan interval allocation.
class ArenaMapper {
 public:
  /// @param alignment The byte alignment for memory addresses (default 32 for AVX/SIMD).
  explicit ArenaMapper(diagnostics::DiagnosticsEngine* diags = nullptr, 
                       size_t alignment = 32)
      : diags_(diags), alignment_(alignment) {}

  ~ArenaMapper() = default;
  ArenaMapper(const ArenaMapper&) = delete;
  ArenaMapper& operator=(const ArenaMapper&) = delete;

  /// @brief Executes the liveness analysis and offset assignment.
  /// @param block A strictly verified, DCE-cleaned block of SIR operations.
  /// @return The compiled memory layout, or an error if allocation fails.
  [[nodiscard]] std::expected<ArenaLayout, MapperError> Run(sir::Block& block);

 private:
  struct LiveInterval {
    const sir::Value* value;
    size_t start_tick;
    size_t end_tick;
    size_t size_bytes;
  };

  /// @brief Calculates the byte size of a tensor, including padding for alignment.
  [[nodiscard]] std::expected<size_t, MapperError> ComputeAlignedSize(
      const sir::Value* value) const;

  /// @brief Performs a forward pass to determine the topological birth and death of each tensor.
  [[nodiscard]] std::expected<std::vector<LiveInterval>, MapperError> ComputeLiveness(
      sir::Block& block) const;

  diagnostics::DiagnosticsEngine* diags_;
  size_t alignment_;
};

}  // namespace seecpp::middle_end::memory

#endif  // SEECPP_MIDDLE_END_MEMORY_ARENA_MAPPER_H_
