#include "src/memory/offset_binder.h"
#include "include/utility/logger.h"

// Assuming your framework provides the core IR definitions and the ArenaLayout
#include "seecpp/sir/sir.h"
#include "seecpp/middle_end/arena_layout.h"

#include <format>
#include <optional>

namespace seecpp::backend {

std::expected<void, CodegenError>
OffsetBinder::run(sir::Block& block, const middle_end::ArenaLayout& layout) {
    bound_operations_ = 0;

    utility::Logger::info("OffsetBinder: Starting memory offset resolution pass");

    // Capture errors from within the lambda during the IR traversal walk
    std::expected<void, CodegenError> pass_result = {};

    block.walk([&](sir::Operation* op) {
        // Short-circuit if an error was previously encountered
        if (!pass_result) return;

        if (auto res = bindOperation(op, layout); !res) {
            pass_result = res;
        }
    });

    // Bubble up any errors caught during the graph traversal
    if (!pass_result) {
        return pass_result;
    }

    utility::Logger::info(std::format(
        "OffsetBinder: Successfully resolved memory offsets for {} operation(s)", 
        bound_operations_
    ));

    return {};
}

std::expected<void, CodegenError>
OffsetBinder::bindOperation(sir::Operation* op, const middle_end::ArenaLayout& layout) {
    std::vector<int64_t> input_offsets;
    std::vector<int64_t> output_offsets;

    // --- 1. Bind Operand (Input) Offsets ---
    for (size_t i = 0; i < op->numOperands(); ++i) {
        const std::string tensor_id = op->operand(i)->id();
        std::optional<uint64_t> offset = layout.getOffset(tensor_id);

        if (!offset.has_value()) {
            return std::unexpected(CodegenError{
                "offset_binding",
                std::format("Failed to resolve memory offset for input operand '{}' in operation '{}'", 
                            tensor_id, op->mnemonic())
            });
        }

        // Cast to int64_t for IR attribute compatibility
        input_offsets.push_back(static_cast<int64_t>(offset.value()));
    }

    // --- 2. Bind Result (Output) Offsets ---
    for (size_t i = 0; i < op->numResults(); ++i) {
        const std::string tensor_id = op->result(i)->id();
        std::optional<uint64_t> offset = layout.getOffset(tensor_id);

        if (!offset.has_value()) {
            return std::unexpected(CodegenError{
                "offset_binding",
                std::format("Failed to resolve memory offset for output result '{}' in operation '{}'", 
                            tensor_id, op->mnemonic())
            });
        }

        output_offsets.push_back(static_cast<int64_t>(offset.value()));
    }

    // --- 3. Attach Resolved Offsets to the Operation ---
    // The final Binary Serializer will read these vectors to pack the execution structs.
    op->setAttribute("input_offsets", input_offsets);
    op->setAttribute("output_offsets", output_offsets);

    ++bound_operations_;
    return {};
}

} // namespace seecpp::backend
