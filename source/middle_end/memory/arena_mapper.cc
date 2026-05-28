#include "source/middle_end/memory/arena_mapper.h"

#include <algorithm>
#include <format>
#include <numeric>

#include "include/utility/logger.hpp"

namespace seecpp::middle_end::memory {

std::expected<ArenaLayout, MapperError> ArenaMapper::Run(sir::Block& block) {
  utility::Logger::info("ArenaMapper: Starting workspace memory allocation.");

  // 1. Calculate the lifespan and size of every intermediate tensor.
  auto intervals_expected = ComputeLiveness(block);
  if (!intervals_expected) {
    return std::unexpected(intervals_expected.error());
  }
  std::vector<LiveInterval> intervals = std::move(intervals_expected.value());

  // 2. Sort intervals by start_tick to simulate linear execution time.
  std::sort(intervals.begin(), intervals.end(),
            [](const LiveInterval& a, const LiveInterval& b) {
              return a.start_tick < b.start_tick;
            });

  ArenaLayout layout;
  
  // Represents active memory blocks: {offset, offset + size, end_tick}
  struct ActiveBlock {
    size_t start_offset;
    size_t end_offset;
    size_t free_after_tick;
  };
  std::vector<ActiveBlock> active_blocks;

  // 3. Linear Scan Allocation (First-Fit approach)
  for (const auto& interval : intervals) {
    // Reclaim memory from tensors whose lifespans have ended.
    std::erase_if(active_blocks, [&](const ActiveBlock& ab) {
      return ab.free_after_tick <= interval.start_tick;
    });

    // Sort active blocks by offset to find the lowest available gap.
    std::sort(active_blocks.begin(), active_blocks.end(),
              [](const ActiveBlock& a, const ActiveBlock& b) {
                return a.start_offset < b.start_offset;
              });

    size_t current_offset = 0;
    bool placed = false;

    // Search for a gap large enough to fit the new tensor.
    for (const auto& ab : active_blocks) {
      if (current_offset + interval.size_bytes <= ab.start_offset) {
        // Gap found!
        placed = true;
        break;
      }
      current_offset = ab.end_offset; // Move past this block
    }

    // Assign the offset and record it in the layout.
    layout.mappings[interval.value] = {
        .offset_bytes = current_offset,
        .size_bytes = interval.size_bytes
    };

    active_blocks.push_back({
        .start_offset = current_offset,
        .end_offset = current_offset + interval.size_bytes,
        .free_after_tick = interval.end_tick
    });

    // Track the high-water mark (the total peak memory required).
    layout.total_arena_size_bytes = std::max(
        layout.total_arena_size_bytes, 
        current_offset + interval.size_bytes
    );
  }

  utility::Logger::info(std::format(
      "ArenaMapper: Allocation complete. Peak workspace size: {} bytes.",
      layout.total_arena_size_bytes));

  return layout;
}

std::expected<std::vector<ArenaMapper::LiveInterval>, MapperError> 
ArenaMapper::ComputeLiveness(sir::Block& block) const {
  std::vector<LiveInterval> intervals;
  std::unordered_map<const sir::Value*, size_t> birth_ticks;
  std::unordered_map<const sir::Value*, size_t> death_ticks;

  size_t tick = 0;
  block.walk([&](sir::Operation* op) {
    // Record the birth tick of all values produced by this operation.
    for (size_t i = 0; i < op->numResults(); ++i) {
      sir::Value* result = op->result(i);
      birth_ticks[result] = tick;
      death_ticks[result] = tick; // Initialize death to birth
    }

    // Extend the death tick of all operands used by this operation.
    for (size_t i = 0; i < op->numOperands(); ++i) {
      sir::Value* operand = op->operand(i);
      if (death_ticks.contains(operand)) {
        death_ticks[operand] = tick;
      }
    }
    tick++;
  });

  // Construct the final intervals with computed sizes
  for (const auto& [value, start] : birth_ticks) {
    auto size_or_err = ComputeAlignedSize(value);
    if (!size_or_err) return std::unexpected(size_or_err.error());

    intervals.push_back({
        .value = value,
        .start_tick = start,
        .end_tick = death_ticks[value],
        .size_bytes = size_or_err.value()
    });
  }

  return intervals;
}

std::expected<size_t, MapperError> ArenaMapper::ComputeAlignedSize(
    const sir::Value* value) const {
  const auto& dims = value->shape().dims;
  
  // The mapper requires static shapes to calculate memory ahead-of-time.
  for (int64_t dim : dims) {
    if (dim == sir::Shape::kDynamic) {
      if (diags_) {
        // Report an error via the user's diagnostics engine
        diags_->Report(diagnostics::Level::Fatal) 
            << "ArenaMapper requires static shapes. Found kDynamic.";
      }
      return std::unexpected(MapperError::kDynamicShapeNotSupported);
    }
  }

  // Multiply dimensions to get total element count
  size_t element_count = std::accumulate(dims.begin(), dims.end(), 1ULL, 
                                         std::multiplies<size_t>());
  
  // Assuming 4 bytes per float (extend this based on value->dtype())
  size_t raw_size = element_count * sizeof(float);

  // Round up to nearest alignment boundary (e.g., nearest 32 bytes)
  size_t aligned_size = (raw_size + alignment_ - 1) & ~(alignment_ - 1);
  return aligned_size;
}

}  // namespace seecpp::middle_end::memory
