#ifndef SEECPP_UTILITY_WEIGHT_BUFFER_H_
#define SEECPP_UTILITY_WEIGHT_BUFFER_H_

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace seecpp::utility {

// Google Style: Enumerators use 'k' prefix for constants.
enum class BufferDtype : uint8_t {
    kF16, kBF16, kF32, kF64,
    kI8,  kI32,  kI64,
    kUnknown
};

struct WeightDescriptor {
    size_t num_elements = 0;
    size_t byte_width = 0;
    BufferDtype dtype = BufferDtype::kUnknown;

    size_t TotalBytes() const { return num_elements * byte_width; }
};

/// @brief Owns binary weight blobs, providing stable pointers via heap allocation.
/// 
/// The WeightBuffer guarantees pointer stability: blobs are stored in 
/// individual unique_ptrs, so map rehashes never invalidate existing data pointers.
class WeightBuffer {
 public:
    WeightBuffer() = default;

    // Blobs are large and owned uniquely; disable copying.
    WeightBuffer(const WeightBuffer&) = delete;
    WeightBuffer& operator=(const WeightBuffer&) = delete;
    WeightBuffer(WeightBuffer&&) = default;
    WeightBuffer& operator=(WeightBuffer&&) = default;

    /// @brief Stores a typed weight blob. 
    /// If 'name' exists, returns the existing span; no silent overwrites.
    template <typename T>
    std::span<const T> Add(std::string_view name,
                           std::span<const T> data,
                           BufferDtype dtype = BufferDtype::kUnknown) {
        const std::string key(name);

        if (auto it = storage_.find(key); it != storage_.end()) {
            return std::span<const T>(
                reinterpret_cast<const T*>(it->second.blob.get()),
                it->second.desc.num_elements);
        }

        const size_t bytes = data.size() * sizeof(T);
        auto blob = std::make_unique<uint8_t[]>(bytes);
        std::memcpy(blob.get(), data.data(), bytes);

        WeightDescriptor desc{data.size(), sizeof(T), dtype};
        const T* ptr = reinterpret_cast<const T*>(blob.get());
        
        storage_.emplace(key, Entry{std::move(blob), desc});
        return std::span<const T>(ptr, data.size());
    }

    /// @brief Typed retrieval. Returns std::nullopt if the name is not present.
    template <typename T>
    std::optional<std::span<const T>> Get(std::string_view name) const {
        auto it = storage_.find(std::string(name));
        if (it == storage_.end()) return std::nullopt;
        
        return std::span<const T>(
            reinterpret_cast<const T*>(it->second.blob.get()),
            it->second.desc.num_elements);
    }

    /// @brief Raw byte access, explicitly named for the WeightPacker's usage.
    std::optional<std::span<const uint8_t>> GetRawBytes(std::string_view name) const {
        auto it = storage_.find(std::string(name));
        if (it == storage_.end()) return std::nullopt;
        
        const auto& e = it->second;
        return std::span<const uint8_t>(e.blob.get(), e.desc.TotalBytes());
    }

    bool Contains(std::string_view name) const {
        return storage_.find(std::string(name)) != storage_.end();
    }

    size_t Count() const { return storage_.size(); }
    
    void Remove(std::string_view name) { storage_.erase(std::string(name)); }

 private:
    struct Entry {
        std::unique_ptr<uint8_t[]> blob;
        WeightDescriptor desc;
    };

    std::unordered_map<std::string, Entry> storage_;
};

} // namespace seecpp::utility

#endif // SEECPP_UTILITY_WEIGHT_BUFFER_H_
