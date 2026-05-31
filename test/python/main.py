import os
import sys
import numpy as np

# Assuming you compile your C++ project with Pybind11 into a module named 'seecpp'
try:
    import seecpp
except ImportError:
    print("[ERROR] seecpp module not found. Did you build the Python bindings?")
    sys.exit(1)

def main():
    print("======================================================")
    print("[SeeC++] End-to-End AOT Python Integration Test")
    print("======================================================\n")

    # =========================================================================
    # PHASE 1: The Compiler (Python Frontend)
    # =========================================================================
    print(">>> Phase 1: Compilation (Python Frontend)")

    # 1. Build the Intermediate Representation
    block = seecpp.sir.Block()
    # Mocking the Python builder API
    # input_tensor = block.add_argument(shape=[1, 4], dtype=seecpp.DType.Float32)
    # ... build MatMul and ReLU ...

    # 2. Configure Compilation Target
    options = seecpp.backend.CodegenOptions()
    options.output_dir = "./build/artifacts"
    options.emit_multithreaded = True

    print(" -> Compiling SIR graph to .see binary...")
    
    # Python gracefully catches C++ std::expected errors as exceptions via Pybind11
    try:
        seecpp.backend.CodegenDriver.compile(block, options)
    except RuntimeError as e:
        print(f"[ERROR] Compilation failed: {e}")
        sys.exit(1)

    binary_path = os.path.join(options.output_dir, "model.see")
    print(f" -> Successfully generated zero-deserialization binary: {binary_path}\n")

    # =========================================================================
    # PHASE 2: The Runtime (Python Inference)
    # =========================================================================
    print(">>> Phase 2: VM Execution")

    # 1. Boot the VM
    engine = seecpp.runtime.RuntimeEngine()
    
    # 2. Memory-map the binary
    print(" -> Mmapping .see file and allocating 64-byte padded arena...")
    try:
        engine.load(binary_path)
    except RuntimeError as e:
        print(f"[ERROR] Runtime load failed: {e}")
        sys.exit(1)

    # 3. Prepare real Numpy Data
    # We must explicitly use float32 to match the C++ backend
    input_data = np.array([-1.5, 3.0, -0.2, 8.4], dtype=np.float32)
    
    print(" -> Injecting input tensors from numpy...")
    # This directly memcpys the numpy buffer into the aligned C++ arena
    engine.set_input(input_data)

    # 4. Fire Execution Loop
    print(" -> Invoking hardware kernels (AVX-512 / NEON)...")
    try:
        engine.invoke()
    except RuntimeError as e:
        print(f"[ERROR] Execution failed: {e}")
        sys.exit(1)

    # 5. Retrieve Outputs
    # We pass the offset (64) and the expected shape to reconstruct a numpy view
    expected_output_offset = 64
    
    # get_output returns a zero-copy numpy view pointing directly into the C++ arena
    output_view = engine.get_output(expected_output_offset, shape=(4,))
    
    print("\n[SUCCESS] Final Evaluated Output:")
    for i in range(len(output_view)):
        print(f"    out[{i}] = {output_view[i]:.4f}")

    print("\n======================================================")
    print("Python System Verified. Ready for production.")
    print("======================================================")

if __name__ == "__main__":
    main()
