#ifndef SEECPP_RUNTIME_ENGINE_H_
#define SEECPP_RUNTIME_ENGINE_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace seecpp::runtime {

struct RuntimeError {
    std::string message;
};

/// @brief The ultra-fast virtual machine for executing .see binaries.
class RuntimeEngine {
 public:
    RuntimeEngine() = default;
    ~RuntimeEngine();

    // Prevent copying to strictly manage the mmap and arena lifecycles
    RuntimeEngine(const RuntimeEngine&) = delete;
    RuntimeEngine& operator=(const RuntimeEngine&) = delete;

    /// @brief Memory-maps the binary and allocates the required dynamic arena.
    [[nodiscard]] std::expected<void, RuntimeError> Load(std::string_view file_path);

    /// @brief Injects the user's raw input data into the start of the memory arena.
    [[nodiscard]] std::expected<void, RuntimeError> SetInput(const float* data, size_t num_elements);

    /// @brief Executes the compiled neural network.
    [[nodiscard]] std::expected<void, RuntimeError> Invoke();

    /// @brief Retrieves a pointer to the final output in the memory arena.
    [[nodiscard]] const float* GetOutput(size_t offset) const;

 private:
    // Memory mapped file state
    int fd_ = -1;
    size_t file_size_ = 0;
    const uint8_t* mmap_ptr_ = nullptr;

    // Dynamic execution memory
    uint8_t* arena_ = nullptr;
    size_t arena_size_ = 0;
};

}  // namespace seecpp::runtime

#endif  // SEECPP_RUNTIME_ENGINE_H_
