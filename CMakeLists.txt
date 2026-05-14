cmake_minimum_required(VERSION 3.15)
project(SeeCompilerPlusPlus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(Protobuf REQUIRED)
find_package(ONNX REQUIRED)

# --- Frontend Library ---
add_library(seecpp_frontend
    source/frontend/onnx_ingressor.cpp
    source/frontend/validator.cpp
    source/frontend/shape_inference.cpp
)

target_include_directories(seecpp_frontend PUBLIC include)
target_link_libraries(seecpp_frontend PUBLIC ONNX::onnx Protobuf::libprotobuf)

# --- Main Driver Executable ---
add_executable(seecpp-frontend source/frontend/frontend_driver.cpp)
target_link_libraries(seecpp-frontend PRIVATE seecpp_frontend)
