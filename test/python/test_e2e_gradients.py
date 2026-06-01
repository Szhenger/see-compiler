# test/python/test_e2e_gradients.py
import os
import subprocess
import tempfile
import torch
import torch.nn as nn
import numpy as np
import numpy.testing as npt
import pytest

# --- 1. The Mathematical Fixtures ---

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Deterministic seed for reproducible testing
        torch.manual_seed(42)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        return self.fc(x)

# --- 2. The Testing Harness ---

class SeeCppHarness:
    """Orchestrates the black-box execution of the SeeC++ compiler."""
    
    def __init__(self, compiler_path: str = "./build/seecpp"):
        self.compiler_path = compiler_path

    def compile(self, onnx_path: str, output_path: str) -> None:
        """Invokes the CLI compiler and asserts zero exit code."""
        result = subprocess.run(
            [self.compiler_path, "--input", onnx_path, "--output", output_path],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Compilation failed: {result.stderr}"

    def execute_and_fetch_gradients(self, see_binary_path: str, input_tensor: np.ndarray) -> dict:
        """
        Placeholder: Executes the compiled binary and extracts the gradient arrays.
        In a real scenario, this might use a lightweight Pybind11 wrapper or read 
        from a generated binary output file.
        """
        # ... execution logic ...
        return {"fc.weight.grad": np.zeros((64, 128))} # Stubbed output

# --- 3. The Integration Tests ---

@pytest.fixture
def harness():
    return SeeCppHarness()

def test_linear_adjoint_derivatives(harness):
    model = SimpleLinearModel()
    dummy_input = torch.randn(1, 128, requires_grad=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "linear.onnx")
        see_path = os.path.join(tmpdir, "linear.see")

        # 1. Export standard ONNX
        torch.onnx.export(model, dummy_input, onnx_path)

        # 2. Generate Golden PyTorch Gradients
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        golden_grad_w = model.fc.weight.grad.detach().numpy()

        # 3. AOT Compile with SeeC++
        harness.compile(onnx_path, see_path)

        # 4. Execute SeeC++ hardware logic
        # (Assuming the input is passed to the compiled model)
        seecpp_grads = harness.execute_and_fetch_gradients(see_path, dummy_input.detach().numpy())
        seecpp_grad_w = seecpp_grads["fc.weight.grad"]

        # 5. The Moment of Truth: Strict Numerical Validation
        npt.assert_allclose(
            seecpp_grad_w, 
            golden_grad_w, 
            rtol=1e-5, 
            atol=1e-5,
            err_msg="SeeC++ Adjoint Graph diverged from PyTorch Calculus."
        )
