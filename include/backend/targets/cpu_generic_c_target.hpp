// include/backend/targets/cpu_generic_c_target.hpp
#pragma once

#include "backend/codegen.hpp" // Contains ICodegenTarget, CodegenOptions
#include <filesystem>
#include <string>

namespace seecpp::ir { class SIRGraph; }
namespace seecpp::backend { class BufferAllocationMap; }

namespace seecpp::backend {

/// Emits highly portable, scalar C/C++ code.
/// Relies on the host compiler (gcc/clang) for auto-vectorization.
class CpuGenericCTarget : public ICodegenTarget {
public:
    explicit CpuGenericCTarget(std::filesystem::path output_dir, CodegenOptions options);
    ~CpuGenericCTarget() override = default;

    // The primary entry point called by the CodegenDriver
    void emit(const ir::SIRGraph& graph, const BufferAllocationMap& arena) override;

protected:
    // Memory constraints for scalar operations
    static constexpr size_t kAlignmentBytes = alignof(float); 

    // Micro-Kernel Generators
    // Returns a string containing the C++ AST/Code for a specific operation
    [[nodiscard]] virtual std::string emitGEMM(const ir::Node& node);
    [[nodiscard]] virtual std::string emitElementwise(const ir::Node& node);
    
private:
    std::filesystem::path output_dir_;
    CodegenOptions options_;
    
    void writeHeaderFile(const BufferAllocationMap& arena);
    void writeSourceFile(const ir::SIRGraph& graph);
};

} // namespace seecpp::backend
