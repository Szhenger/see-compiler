# SeeC++: An AOT ONNX-to-C++ Training Compiler
### *Where the Rigor of Mathematical Analysis Meets the Metal of Performance Engineering*

[![Standard: ONNX](https://img.shields.io/badge/Standard-ONNX%20v1.16-blue)](https://onnx.ai/)
[![Backend: C++20](https://img.shields.io/badge/Backend-C%2B%2B20-00599C)](https://isocpp.org/)

**SeeC++** is a specialized Ahead-of-Time (AOT) compiler that transforms standard ONNX (Open Neural Network Exchange) computation graphs into standalone, dependency-free `C++` source code. 

## The "Why": A Love for Mathematical Analysis
`SeeC++` was born at the intersection of a engineering pivot with **Mathematical Analysis**. 

In college, `Real Analysis` was my favorite subject — the study of real-valued functions. While building `SeeC`, I realized that the "Leaky Abstractions" of modern ML (Machine Learning) frameworks often obscure the beauty of the underlying calculus. I am developing `SeeC++` to reclaim that transparency: translating the abstract **Chain Rule** directly into **AVX-512** instructions.

---

## The "Vision": A Passion for Mechanical Sympathy
Most Deep Learning frameworks act as "interpreters," dispatching kernels one-by-one with significant overhead. **SeeC++** treats a neural network as a fixed, statically analyzable system of equations.

* **No Python Runtime:** The training loop is compiled into a single `C++` translation unit.
* **Zero-Allocation:** Using the principle of "Compact Support," we map every tensor to a static byte-offset in a pre-allocated **Memory Arena**.
* **Pure Calculus:** Every gradient is derived through symbolic Automatic Differentiation at compile-time.

---

## Mathematical Foundation

In `SeeC++`, we represent a Neural Network as a composite function $F \ : \ \mathbb{R}^n \to \mathbb{R}^m$ where $F = f_L \circ f_{L-1} \circ \dots \circ f_1$.

### 1. The Adjoint Equation
To train the model, `SeeC++` constructs the **Adjoint Graph**. By applying the Chain Rule in reverse topological order, the compiler generates the gradient code for every weight $w_i$:

$$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial f_L} \cdot \frac{\partial f_L}{\partial f_{L-1}} \cdot \dots \cdot \frac{\partial f_i}{\partial w_i}$$

### 2. Kernel Fusion & Function Composition
In `SeeC++`, we fuse operations like `Add + ReLU + Dropout` into a single loop, ensuring that data stays in the `L1/L2` cache and avoids unnecessary round-trips to `DRAM`.

---

## System Architecture

### Module I: Ingress (Frontend)
* **Input:** Industry-standard `.onnx` files.
* **Process:** Parses the Protobuf binary into the **SeeC++ Intermediate Representation (SIR)**.

### Module II: The Analytic Pass (Middle-End)
* **Autodiff Engine:** The "Calculus Core" that symbolically appends gradient nodes to the forward graph.
* **Static Memory Mapper:** Pre-calculates the exact lifespan of every tensor to minimize the memory footprint.

### Module III: The System Pass (Backend)
* **Emitter:** Translates the `SIR` into high-performance `C++20`.
* **Optimization:** Inlines hardware intrinsics for `x86_64` (AVX-512) and `ARM` (NEON).

---

## Repository Structure & Build System

**SeeC++** adheres to the **Google C++ Style Guide** and utilizes modern **CMake** to ensure mechanical sympathy, structural predictability, and robust cross-platform compilation. 

```text
see-compiler/
├── README.md               # Specification
├── .clang-format           # Google-style C++ enforcement
├── .clang-tidy             # Google-style C++ adjustment
├── CMakeLists.txt          # Master build orchestrator
├── cmake/                  # Custom CMake modules (e.g., Protobuf codegen)
├── include/
│   └── seecpp/             # Protected public API boundary (compiler.h, options.h)
├── source/                 # Internal implementation (headers co-located with source)
│   ├── frontend/
│   ├── middle_end/
│   └── backend/
├── test/                   
│   ├── cpp/                # High-performance GoogleTest unit tests
│   ├── python/             # End-to-end pytest framework integration
│   └── fixtures/           # Deterministic .onnx mathematical scenarios
├── proto/
│   └── sir.proto           # SeeC++ Intermediate Representation schema
├── external/               # External vendor dependencies (ONNX, GoogleTest) managed via CMake
├── docs/                   # Architectural Decision Records (ADRs) and Calculus Proofs
├── tools/                  # Developer ergonomics (format.sh, lint.sh)
└── .github/workflows/      # Continuous Integration pipelines
