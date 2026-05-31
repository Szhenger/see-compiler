cmake_minimum_required(VERSION 3.20)
project(seecpp VERSION 1.0.0 LANGUAGES CXX)

# ==============================================================================
# Global Configuration
# ==============================================================================

# Enforce C++20 for std::expected, std::span, and modern memory features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Treat warnings as errors to enforce strict memory safety
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
elseif(MSVC)
    add_compile_options(/W4 /WX)
endif()

# ==============================================================================
# Target 1: The Compiler (seecpp_compiler)
# Heavyweight target used only during Ahead-of-Time compilation.
# ==============================================================================

add_library(seecpp_compiler STATIC
    src/memory/offset_binder.cc
    src/serialization/weight_packer.cc
    src/backend/codegen_driver.cc
)

# Expose the public headers to consumers, but keep src/ private
target_include_directories(seecpp_compiler
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ==============================================================================
# Target 2: The Execution Engine (seecpp_runtime)
# Ultra-lightweight target deployed to the edge/server. Zero compiler dependencies.
# ==============================================================================

add_library(seecpp_runtime STATIC
    src/runtime/runtime_engine.cc
    src/runtime/avx512_kernels.cc
    src/runtime/neon_kernels.cc
)

target_include_directories(seecpp_runtime
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Hardware-Specific SIMD Tuning ---
# We surgically apply architecture flags ONLY to the kernel files. 
# This prevents the compiler from accidentally auto-vectorizing generic 
# setup code and causing illegal instruction faults on unsupported hardware.

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # AVX-512 Kernels (x86_64)
    set_source_files_properties(src/runtime/avx512_kernels.cc 
        PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512vl -mavx512bw -mavx512dq -O3"
    )
    
    # NEON Kernels (ARM64)
    # Note: On Apple Silicon (M1/M2/M3), NEON is enabled by default, 
    # but explicit tagging is good practice for cross-compilation.
    set_source_files_properties(src/runtime/neon_kernels.cc 
        PROPERTIES COMPILE_FLAGS "-march=armv8-a+simd -O3"
    )
elseif(MSVC)
    set_source_files_properties(src/runtime/avx512_kernels.cc 
        PROPERTIES COMPILE_FLAGS "/arch:AVX512 /O2"
    )
endif()

# ==============================================================================
# Optional: End-to-End Integration Test
# ==============================================================================
# To build the test stub, run: cmake -DSEECPP_BUILD_TESTS=ON ..

option(SEECPP_BUILD_TESTS "Build the end-to-end integration stub" OFF)

if(SEECPP_BUILD_TESTS)
    add_executable(seecpp_integration_test tests/main.cc)
    
    # The test needs both the compiler (to generate .see) and runtime (to execute it)
    target_link_libraries(seecpp_integration_test PRIVATE 
        seecpp_compiler 
        seecpp_runtime
    )
endif()
