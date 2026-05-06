cmake_minimum_required(VERSION 3.15)
project(SeeCompilerPlusPlus LANGUAGES CXX)

# Use C++17 or C++20 for modern compiler development
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find Dependencies (Ensure ONNX and Protobuf are installed)
find_package(ONNX REQUIRED)
find_package(Protobuf REQUIRED)

# 2. Define the Executable
# This 'seec' binary is what your Python 'runner' fixture will call
add_executable(seec frontend_driver.cpp)

# 3. Link Libraries
# You likely need the ONNX library and its protobuf backing
target_link_libraries(seec PRIVATE onnx onnx_proto)

# 4. Include Directories
target_include_directories(seec PRIVATE ${ONNX_INCLUDE_DIRS})

# 5. Output Management (The "Test Bridge")
# Force the binary into a predictable 'bin' folder so your 
# Python runner doesn't have to guess based on build type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)