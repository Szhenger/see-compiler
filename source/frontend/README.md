# SeeC++ Frontend: The Semantic Ingress & Schema

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

The Frontend is divided into two major functional units: the **Ingress** and the **Parser/Validator**.

### A. The Ingress (Protobuf Deserializer)
The Ingress is the physical interface with the `.onnx` binary.
* **Responsibility:** Low-level extraction of the `ModelProto`.
* **Weight Extraction:** It separates "Initializers" (static weights) from the graph structure, preparing them for the **Static Memory Arena**.
* **Attribute Mapping:** Converts ONNX node attributes (like `axis`, `alpha`, or `strides`) from Protobuf types into native C++ variants.

### B. The Parser (Graph Reconstruction)
The Parser transforms the flat list of ONNX nodes into a **Directed Acyclic Graph (DAG)**.
* **Topological Sorting:** It reorders nodes to ensure that no operation is executed before its dependencies are available.
* **Symbol Table Resolution:** It maps string-based tensor names (e.g., "input.1") to unique `TensorNode` pointers in the SIR.
* **Shape Inference Engine:** Since many ONNX models do not explicitly state the shape of every intermediate activation, the Parser propagates shapes from the input $X$ through the entire network using operator-specific logic.

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

## 4. Semantic Validation Rules
Before passing the graph to the Middle-End, the Frontend enforces **Real Analysis invariants**:
1.  **Rank Consistency:** Operators like `Conv2D` require a specific rank (typically 4: $NCHW$).
2.  **Broadcast Compatibility:** Validates that shapes are compatible for element-wise operations (e.g., adding a scalar to a matrix).
3.  **Differentiability Check:** Ensures all operators in the graph have a defined derivative in our **Autodiff Library**.

---

## 5. Summary of the Ingress Flow
1.  **Load:** Binary Protobuf $\to$ `ModelProto`.
2.  **Extract:** Weights $\to$ `Static Weight Map`.
3.  **Reconstruct:** Nodes $\to$ `SIR::Graph`.
4.  **Infer:** Propagate shapes $\to$ `SIR::TensorMetadata`.
5.  **Verify:** Semantic Check $\to$ **Ready for Middle-End**.
