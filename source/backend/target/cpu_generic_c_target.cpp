// source/backend/targets/cpu_generic_c_target.cpp
#include "include/backend/targets/cpu_generic_c_target.hpp"
#include "include/utility/logger.hpp"

#include <fstream>
#include <format>
#include <stdexcept>

namespace seecpp::backend {

CpuGenericCTarget::CpuGenericCTarget(std::filesystem::path output_dir, CodegenOptions options)
    : output_dir_(std::move(output_dir)), options_(std::move(options)) {}

void CpuGenericCTarget::emit(const ir::SIRGraph& graph, const BufferAllocationMap& arena) {
    utility::Logger::info("Emitting generic C/C++ target code.");
    
    // 1. Write the Memory Arena structure and function signatures
    writeHeaderFile(arena);
    
    // 2. Iterate through the computation graph and emit the kernels
    writeSourceFile(graph);
}

void CpuGenericCTarget::writeHeaderFile(const BufferAllocationMap& arena) {
    std::filesystem::path header_path = output_dir_ / "model.h";
    std::ofstream out(header_path);
    if (!out) throw std::runtime_error("Failed to open model.h for writing.");

    // Note the use of raw string literals to keep emitted code clean
    out << std::format(R"(
#pragma once
#include <cstdint>
#include <cstddef>

// SeeC++ Auto-Generated Header
// Target: Generic C++ (Scalar)
// Arena Size: {} bytes

alignas({ /* alignment */ }) struct MemoryArena {{
    uint8_t buffer[{}];
}};

// Entry Points
void seecpp_forward(MemoryArena* arena);
void seecpp_backward(MemoryArena* arena);
)", arena.total_bytes(), kAlignmentBytes, arena.total_bytes());
}

std::string CpuGenericCTarget::emitElementwise(const ir::Node& node) {
    // A naive scalar loop. The host compiler (gcc/clang) might auto-vectorize this, 
    // but we offer no guarantees.
    return std::format(R"(
    // Node: {}
    for (size_t i = 0; i < {}; ++i) {{
        arena->buffer[{}] = arena->buffer[{}] + arena->buffer[{}]; // Example: Add
    }}
)", node.name(), node.size(), node.out_offset(), node.in1_offset(), node.in2_offset());
}

// ... file writing logic for writeSourceFile omitted for brevity ...

} // namespace seecpp::backend
