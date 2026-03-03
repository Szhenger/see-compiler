#pragma once

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

// Dtype tag stored alongside each blob so the backend can verify
// it is casting to the correct type at retrieval time.
enum class BufferDtype : uint8_t {
    F16, BF16, F32, F64,
    I8,  I32,  I64,
    Unknown
};

/// Metadata descriptor stored alongside each weight blob.
struct WeightDescriptor {
    size_t      num_elements = 0;
    size_t      byte_width   = 0;   // sizeof(T) at the time of insertion
    BufferDtype dtype        = BufferDtype::Unknown;

    size_t totalBytes() const { return num_elements * byte_width; }
};

/// Owns large binary weight blobs separately from the SIR graph.
///
/// Pointer stability guarantee:
///   Pointers returned by add() and get() remain valid for the lifetime
///   of the WeightBuffer, regardless of how many subsequent add() calls
///   are made. This holds because each blob is individually heap-allocated
///   via unique_ptr<uint8_t[]> and the map stores owning handles, not
///   inline data.
///
/// Duplicate names: add() on an already-present name is a no-op that
/// returns the existing descriptor. Call remove() first to replace.
class WeightBuffer {
public:
    WeightBuffer() = default;

    // Non-copyable — blobs are large; copies must be explicit.
    WeightBuffer(const WeightBuffer&)            = delete;
    WeightBuffer& operator=(const WeightBuffer&) = delete;
    WeightBuffer(WeightBuffer&&)                 = default;
    WeightBuffer& operator=(WeightBuffer&&)      = default;

    /// Store a typed weight blob. Returns a stable typed span over the data.
    /// If `name` already exists the call is a no-op and the existing span
    /// is returned — no silent overwrite, no dangling pointer.
    template <typename T>
    std::span<const T> add(std::string_view name,
                           std::span<const T> data,
                           BufferDtype dtype = BufferDtype::Unknown)
    {
        const std::string key(name);

        // Guard: duplicate insert returns existing data untouched.
        if (auto it = storage_.find(key); it != storage_.end())
            return std::span<const T>(
                reinterpret_cast<const T*>(it->second.blob.get()),
                it->second.desc.num_elements);

        const size_t bytes = data.size() * sizeof(T);
        auto blob = std::make_unique<uint8_t[]>(bytes);   // single allocation
        std::memcpy(blob.get(), data.data(), bytes);

        WeightDescriptor desc;
        desc.num_elements = data.size();
        desc.byte_width   = sizeof(T);
        desc.dtype        = dtype;

        const T* ptr = reinterpret_cast<const T*>(blob.get());
        storage_.emplace(key, Entry{std::move(blob), desc});
        return std::span<const T>(ptr, data.size());
    }

    /// Typed retrieval. Returns nullopt if the name is not present.
    template <typename T>
    std::optional<std::span<const T>> get(std::string_view name) const {
        auto it = storage_.find(std::string(name));
        if (it == storage_.end()) return std::nullopt;
        const auto& e = it->second;
        return std::span<const T>(
            reinterpret_cast<const T*>(e.blob.get()),
            e.desc.num_elements);
    }

    /// Raw byte access for the backend serialiser.
    std::optional<std::span<const uint8_t>> getRaw(std::string_view name) const {
        auto it = storage_.find(std::string(name));
        if (it == storage_.end()) return std::nullopt;
        const auto& e = it->second;
        return std::span<const uint8_t>(e.blob.get(), e.desc.totalBytes());
    }

    /// Metadata without touching the blob data.
    std::optional<WeightDescriptor> descriptor(std::string_view name) const {
        auto it = storage_.find(std::string(name));
        if (it == storage_.end()) return std::nullopt;
        return it->second.desc;
    }

    bool        contains(std::string_view name) const {
        return storage_.find(std::string(name)) != storage_.end();
    }
    size_t      count()  const { return storage_.size(); }
    bool        empty()  const { return storage_.empty(); }

    /// Remove a weight by name. Safe to call if name is absent.
    void remove(std::string_view name) { storage_.erase(std::string(name)); }

    /// Iterate all weight names (order unspecified).
    template <typename Fn>
    void forEach(Fn&& fn) const {
        for (const auto& [name, entry] : storage_)
            fn(name, entry.desc);
    }

private:
    struct Entry {
        std::unique_ptr<uint8_t[]> blob;
        WeightDescriptor           desc;
    };

    std::unordered_map<std::string, Entry> storage_;
};

} // namespace seecpp::utility