# SeeC++ Frontend: The Semantic Ingress & Representation Schema

## 1. Mathematical Foundations: The Tensor Manifold

The Frontend does not view data as flat buffers, but as **Ranked Tensors**. To preserve the semantic encoding of the ONNX binary, we define a Tensor $\mathcal{T}$ as a triple:
$$\mathcal{T} = (S, D, \Sigma)$$

Where:
* **$S$ (Shape):** An ordered tuple $(d_1, d_2, \dots, d_n) \in \mathbb{N}^n$ defining the dimensionality.
* **$D$ (Data):** The underlying coordinate space (typically $\mathbb{R}$ or $\mathbb{Z}$).
* **$\Sigma$ (Strides):** The mapping function $\mathbb{N}^n \to \mathbb{N}$ that defines how multidimensional indices translate to linear memory offsets.

### Semantic Preservation vs. Isomorphic Buffers
A tensor of shape $[2, 6]$ and a tensor of shape $[3, 4]$ are **isomorphic** in memory (both contain 12 elements), but they are **semantically distinct** in the world of Linear Algebra. The Frontend’s primary role is to enforce the **Structural Constraints** of the operations (e.g., ensuring inner-product dimensions match in a $MatMul$).

---

## 2. System Architecture: The Ingress Pipeline

The Frontend is divided into four major functional units that progressively lower and sanitize the incoming graph.

### A. The Ingress (Protobuf Deserializer)
The Ingress is the physical interface with the `.onnx` binary.
* **Responsibility:** Low-level extraction of the `ModelProto`.
* **Weight Extraction:** Separates "Initializers" (static weights) from the graph structure, preparing them for the **Static Memory Arena**.
* **Type Bridging:** Immediately maps external `TensorProto::DataType` enums to internal `seecpp::DType` enumerations, preventing third-party standards from bleeding into the Middle-End.
* **Attribute Mapping:** Converts ONNX node attributes (like `axis`, `alpha`, or `strides`) from Protobuf types into native C++ variants.

### B. The Parser (Graph Reconstruction)
The Parser transforms the flat list of ONNX nodes into a **Directed Acyclic Graph (DAG)**.
* **Memory Ownership:** Enforces strict C++20 resource management. A central `Graph` object holds explicit ownership of all nodes via `std::unique_ptr`, while topological dependencies (edges) are represented as non-owning raw pointers to prevent memory leaks.
* **Topological Sorting:** Reorders nodes to ensure that no operation is executed before its dependencies are available.
* **Symbol Table Resolution:** Maps string-based tensor names (e.g., "input.1") to unique `TensorNode` pointers in the SIR.
* **Shape Inference Engine:** Propagates known shapes from the input $X$ through the entire network using operator-specific logic.

### C. The Canonicalizer (Operator Lowering)
* **Responsibility:** Simplifies the massive ONNX operator set (~200 ops) down to a minimal set of ~30 mathematically pure primitives. 
* **Action:** Breaks down composite operators (e.g., de-sugaring `Gemm` into atomic `MatMul`, `Mul`, and `Add` nodes) to drastically reduce the complexity of the downstream Autodiff engine.

### D. The Constant Folder (Partial Evaluation)
* **Responsibility:** Eliminates dead code and static calculations exported by Python tracing.
* **Action:** Acts as a compile-time interpreter, traversing the DAG to execute mathematical operators where all inputs are constants. It replaces these subgraphs with pre-calculated Initializers, preventing the Backend from generating wasted C++ loops.

---

## 3. Schema Definitions: The SIR (SeeC++ IR)

To maintain semantic integrity, the Frontend populates the following schema:

| Component | Mathematical Role | System Implementation |
| :--- | :--- | :--- |
| **TensorNode** | Coordinate Space | Stores Shape, DType, and Stride metadata. |
| **OperatorNode** | Functional Mapping | Defines the transform $f(x)$. |
| **Edge** | Data Flow | Represents the "Adjoint" path for the Autodiff engine. |
| **Initializer** | Constant | Maps static weights to the pre-allocated binary blob. |

---

## 4. Semantic Validation Rules & Diagnostics

Before passing the graph to the Middle-End, the Frontend enforces **Real Analysis invariants** through a centralized **Diagnostics Engine**. Instead of raw C++ exceptions, structural violations trigger formatted, Clang-style terminal traces pinpointing the exact ONNX node and dimensional mismatch.

1. **Rank Consistency:** Operators like `Conv2D` require a specific rank (typically 4: $NCHW$).
2. **Broadcast Compatibility:** Validates that shapes are compatible for element-wise operations (e.g., adding a scalar to a matrix).
3. **Differentiability Check:** Ensures all operators in the graph have a defined derivative in our Autodiff Library.
4. **The AOT Shape Guarantee:** Because `SeeC++` relies on a zero-allocation Static Memory Arena, dynamic shapes (e.g., batch size `-1`) are mathematically incompatible. If the shape inference pass yields any dimension $d_i \notin \mathbb{N}^+$, the compiler aborts compilation.

---

## 5. Summary of the Ingress Flow

1. **Load:** Binary Protobuf $\to$ `ModelProto`.
2. **Extract & Bridge:** Weights & Enums $\to$ `Static Weight Map` & `seecpp::DType`.
3. **Reconstruct:** Nodes $\to$ `SIR::Graph` (Topologically Sorted).
4. **Canonicalize & Fold:** Lower Operators & Pre-compute Static Subgraphs.
5. **Infer:** Propagate shapes $\to$ `SIR::TensorMetadata`.
6. **Validate & Diagnose:** Semantic Check $\to$ **Ready for Middle-End**.

---

## 6. Subrepository Structure & Build System

```text
see_compiler/
├── include/
│   └── seecpp/
│       └── frontend/
│           └── frontend_manager.h    # The ONLY public API for the Frontend
│
├── source/
│   └── frontend/
│       ├── ingress/                  # Stage 1: Protobuf Extraction
│       │   ├── protobuf_reader.h
│       │   ├── protobuf_reader.cc
│       │   ├── type_bridge.h         # Isolates ONNX enums from SeeC++ enums
│       │   └── type_bridge.cc
│       │
│       ├── parser/                   # Stage 2: DAG Reconstruction
│       │   ├── graph_builder.h       # Manages std::unique_ptr ownership of nodes
│       │   ├── graph_builder.cc
│       │   ├── shape_inference.h     # Propagates shapes from inputs to outputs
│       │   └── shape_inference.cc
│       │
│       ├── transforms/               # Stages 3 & 4: Lowering and Evaluation
│       │   ├── canonicalizer.h       # De-sugars complex ONNX ops (e.g., Gemm)
│       │   ├── canonicalizer.cc
│       │   ├── constant_folder.h     # Pre-computes static subgraphs
│       │   └── constant_folder.cc
│       │
│       ├── validator/                # Stage 5: Real Analysis Constraints
│       │   ├── semantic_check.h      # Enforces rank, broadcast, and AOT shape rules
│       │   └── semantic_check.cc
│       │
│       └── diagnostics/              # Stage 6: Error Formatting
│           ├── diagnostics_engine.h  # Clang-style terminal error tracing
│           └── diagnostics_engine.cc
