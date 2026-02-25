#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>

namespace seecpp::utility {

/**
 * @brief Manages large binary blobs (weights/biases) separately from the IR graph.
 */
class WeightBuffer {
    public:
        // Stores a blob of data and returns a stable pointer to it
        template<typename T>
        const T* addWeight(const std::string& name, const std::vector<T>& data) {
            size_t bytes = data.size() * sizeof(T);
            auto blob = std::make_unique<std::vector<uint8_t>>(bytes);
            
            // Copy raw data into our byte-buffer
            std::memcpy(blob->data(), data.data(), bytes);
            
            const T* ptr = reinterpret_cast<const T*>(blob->data());
            storage[name] = std::move(blob);
            return ptr;
        }

        const uint8_t* getRaw(const std::string& name) const {
            if (storage.find(name) == storage.end()) return nullptr;
            return storage.at(name)->data();
        }

    private:
        // Maps weight name to its owned byte-storage
        std::unordered_map<std::string, std::unique_ptr<std::vector<uint8_t>>> storage;
};

} // namespace seecpp::utility