# SeeC++ Testing Strategy & Architecture

This document delineates the testing architecture for the SeeC++ Ahead-of-Time (AOT) compiler. Given our target constraints of high-performance execution on bare silicon, mathematical precision and rigid structural boundaries must be enforced across all compilation phases.

Our verification matrix splits into four distinct domains, each governed by an isolated testing philosophy tailored to its implementation pattern.

---

## 1. Frontend Driver: Fixture-Driven Integration

The `FrontendDriver` (`source/frontend/frontend_driver.cc`) serves as an immutable, short-circuiting orchestrator for the initial ingestion pipeline:

$$\text{Ingest} \longrightarrow \text{Shape Inference} \longrightarrow \text{Validate} \longrightarrow \text{Middle-End} \longrightarrow \text{Serialize}$$

Because sub-components are instantiated directly on the stack within the `Run()` method, traditional object mocking is unavailable. 

### Implementation Requirements
* **The Cascade Gate Pattern:** Tests must utilize hand-crafted, invalid `.onnx` files to intentionally trip internal boundaries (e.g., providing structurally valid graph topologies that fail dimensional validation to assert `ShapeInferencePass` early termination).
* **Global State Isolation:** The frontend couples to a mutable global buffer (`seecpp::utility::global_weight_buffer`). The test fixture **must** clear this state during `TearDown()` to prevent non-deterministic test flakiness during parallel test execution.
* **System IO Assertions:** The driver must be tested against hostile environment states (e.g., unwritable target paths) to prove that the serialization stage handles system friction without throwing unhandled exceptions.

---

## 2. Middle-End Pass Manager: White-Box Mocking

The `PassManager` (`source/middle_end/pass_manager.cc`) manages optimization passes over the graph. It relies entirely on Dependency Injection by accepting polymorphic pointers to abstract `Pass` classes, making it highly testable.

### Implementation Requirements
* **Mutation Flag Verification:** Utilizing `StrictMock<MockPass>`, tests must confirm that the manager accurately tracks changes across the entire optimization pipeline using bitwise OR ($\|=$) accumulation.
* **The Zero-Mutation Optimization:** Tests must explicitly verify that if an optimization pass returns `false` (indicating zero structural modifications), the manager bypasses the computationally heavy `block.Verify()` call.
* **Invariant Short-Circuiting:** If a pass modifies the graph but subsequent validation fails, the manager must immediately abort and return a `PassError::kVerificationFailed` variant, blocking corrupted intermediate representations from progressing.

---

## 3. Backend Codegen Driver: State-Driven Integration

The `CodegenDriver` (`source/backend/codegen_driver.h`) converts abstract graph mathematics into a machine-executable `.see` binary. It runs through a concrete, four-stage stack: `InstructionSelector`, `OffsetBinder`, `WeightPacker`, and `Serializer`.

### Implementation Requirements
* **Phase-Specific Error Mapping:** The driver returns a diagnostic `std::expected<void, CodegenError>` type. The test suite must strictly validate that the `error().phase` string matches the exact point of system failure (e.g., `"instruction_selection"`, `"offset_binding"`).
* **Memory Arena Boundary Tests:** Programmatic `sir::Block` inputs must be fed with out-of-bounds allocation requests to verify that the `OffsetBinder` catches address collisions before binary layout generation.
* **Artifact Validation:** "Happy Path" verification must confirm not only a successful return code but must actively validate filesystem state by asserting that the compiled binary size is non-zero.

---

## 4. End-to-End (E2E) Integration: The Numerical Oracle

While the C++ suites validate architectural stability, the Python suite (`test/python/`) validates mathematical correctness. We leverage PyTorch not as a runtime component, but as an analytical "Golden Reference."

### The Integration Pipeline
1. **Topology Generation:** Define a neural network block in PyTorch and export it to an `.onnx` fixture.
2. **Oracle Evaluation:** Compute the forward pass and execute `.backward()` to capture exact algorithmic gradients.
3. **AOT Compilation:** Invoke the compiled SeeC++ executable via Python `subprocess` to generate the `.see` binary.
4. **Silicon Execution:** Run the SeeC++ engine on native hardware to generate local gradient weights.
5. **Numerical Assertion:** Compare the arrays using `numpy.testing.assert_allclose` under strict tolerances:

$$\text{Tolerance} \le 1 \times 10^{-5}$$

### Evaluation Tiers
* **Linear Baselines:** Asserts simple matrix multiplication and bias accumulation layout ($XW + B$).
* **Non-Differentiable Boundaries:** Stresses subgradient behavior at the sharp threshold ($x = 0$) of the symbolic `ReLU` engine.
* **Fused Operators:** Validates multi-dimensional memory strides and data locality under kernel fusion layouts (e.g., `Conv2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU`).
