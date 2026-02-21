#ifndef SEECPP_FRONTEND_ONNX_INGRESSOR_HPP
#define SEECPP_FRONTEND_ONNX_INGRESSOR_HPP

#include "sir.hpp"
#include <string>
#include <map>

namespace seecpp::frontend {

/**
 * @brief
 */
class OnnxIngressor {
public:
    OnnxIngressor() = default;

    /**
     * @brief Parses the ONNX file and populates the provided SIR Block.
     * @param model_path Path to the .onnx binary file.
     * @param block The SIR Block to populate with sc_high operations.
     * @return true if ingestion was successful.
     */
    bool ingest(const std::string& model_path, sir::Block& block);

private:
    // Tracks the mapping from ONNX tensor names to internal SSA Values
    std::map<std::string, sir::Value*> tensor_map;
    
    // Internal helper to convert ONNX types to SeeC++ types
    sir::DataType mapDataType(int onnx_type);
};

} // namespace seecpp::frontend

#endif // SEECPP_FRONTEND_ONNX_INGRESSOR_HPP