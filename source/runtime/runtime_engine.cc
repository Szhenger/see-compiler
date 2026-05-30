#include "src/runtime/runtime_engine.h"
#include "src/serialization/schema.h"
#include "src/runtime/kernels.h" // the Gemv kernel we built
#include "include/utility/logger.h"

#include <format>
#include <cstdlib>
#include <cstring>

// POSIX Memory Mapping
#include <fcntl.h>
#include <sys/mmap.h>
#include <sys/stat.h>
#include <unistd.h>

namespace seecpp::runtime {

RuntimeEngine::~RuntimeEngine() {
    if (mmap_ptr_ != nullptr && mmap_ptr_ != MAP_FAILED) {
        munmap(const_cast<uint8_t*>(mmap_ptr_), file_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
    if (arena_ != nullptr) {
        free(arena_); // Free the aligned allocation
    }
}

std::expected<void, RuntimeError> RuntimeEngine::Load(std::string_view file_path) {
    // 1. Open the file
    fd_ = open(file_path.data(), O_RDONLY);
    if (fd_ == -1) {
        return std::unexpected(RuntimeError{std::format("Failed to open file: {}", file_path)});
    }

    // 2. Get file size
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        return std::unexpected(RuntimeError{"Failed to stat file."});
    }
    file_size_ = sb.st_size;

    // 3. Memory Map (Zero-copy, directly from disk to RAM)
    mmap_ptr_ = static_cast<const uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (mmap_ptr_ == MAP_FAILED) {
        return std::unexpected(RuntimeError{"Failed to mmap file."});
    }

    // 4. Validate Schema Header
    const auto* header = reinterpret_cast<const backend::FileHeader*>(mmap_ptr_);
    if (header->magic != backend::kSeeMagic || header->version != backend::kCurrentVersion) {
        return std::unexpected(RuntimeError{"Invalid .see file magic bytes or unsupported version."});
    }

    // 5. Allocate the hardware-aligned Memory Arena (Calculated earlier by OffsetBinder)
    arena_size_ = header->arena_size;
    // 64-byte alignment for AVX-512 / cache lines
    arena_ = static_cast<uint8_t*>(std::aligned_alloc(64, arena_size_)); 
    if (!arena_) {
        return std::unexpected(RuntimeError{"Failed to allocate aligned execution arena."});
    }

    utility::Logger::Info(std::format(
        "Runtime: Loaded '{}'. Mapped {} bytes. Allocated Arena: {} bytes.",
        file_path, file_size_, arena_size_
    ));

    return {};
}

std::expected<void, RuntimeError> RuntimeEngine::SetInput(const float* data, size_t num_elements) {
    if (!arena_) return std::unexpected(RuntimeError{"Model not loaded."});
    
    // Assuming the OffsetBinder mapped the primary input tensor to offset 0
    std::memcpy(arena_, data, num_elements * sizeof(float));
    return {};
}

std::expected<void, RuntimeError> RuntimeEngine::Invoke() {
    if (!mmap_ptr_ || !arena_) {
        return std::unexpected(RuntimeError{"Cannot invoke: Model not loaded."});
    }

    const auto* header = reinterpret_cast<const backend::FileHeader*>(mmap_ptr_);
    
    // Resolve absolute pointers from the mmap base
    const auto* instructions = reinterpret_cast<const backend::SerializedInstruction*>(
        mmap_ptr_ + header->text_offset
    );
    const uint8_t* rodata_base = mmap_ptr_ + header->rodata_offset;

    // =========================================================================
    // THE EXECUTION LOOP
    // This entirely replaces your legacy `cpu_code.cpp` logic.
    // =========================================================================
    for (uint64_t i = 0; i < header->text_size; ++i) {
        const auto& inst = instructions[i];

        switch (inst.opcode) {
            // Example Opcode: GEMV (General Matrix-Vector Multiply)
            case 10: { // Assuming 10 is backend::Opcode::kGemv
                // Decode offsets. Weights come from rodata, activations from arena.
                const float* A_weights = reinterpret_cast<const float*>(rodata_base + inst.inputs[0]);
                const float* x_vector  = reinterpret_cast<const float*>(arena_ + inst.inputs[1]);
                const float* bias      = reinterpret_cast<const float*>(rodata_base + inst.inputs[2]);
                float* y_output        = reinterpret_cast<float*>(arena_ + inst.outputs[0]);

                // The shapes would either be baked into the instruction struct, 
                // or read from a tiny metadata header preceding the weight blob.
                // For this example, let's assume they were packed into inputs[3].
                size_t m = inst.inputs[3] >> 32;       // Upper 32 bits
                size_t n = inst.inputs[3] & 0xFFFFFFFF; // Lower 32 bits

                // Dispatch to the AVX-512 or NEON kernel we wrote earlier!
                kernels::Gemv(A_weights, x_vector, bias, y_output, m, n);
                break;
            }

            // Example Opcode: RELU
            case 11: {
                float* data = reinterpret_cast<float*>(arena_ + inst.inputs[0]);
                size_t count = inst.inputs[1];
                for (size_t j = 0; j < count; ++j) {
                    if (data[j] < 0.0f) data[j] = 0.0f;
                }
                break;
            }

            default:
                return std::unexpected(RuntimeError{
                    std::format("Encountered unknown hardware opcode: {}", inst.opcode)
                });
        }
    }

    return {};
}

const float* RuntimeEngine::GetOutput(size_t offset) const {
    if (!arena_ || offset >= arena_size_) return nullptr;
    return reinterpret_cast<const float*>(arena_ + offset);
}

}  // namespace seecpp::runtime
