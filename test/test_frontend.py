import subprocess
import os
import pytest
import onnx
from onnx import helper, TensorProto

# Path to your compiled C++ binary
FRONTEND_BINARY = "../build/source/driver/frontend_driver"

def create_simple_model(path):
    """Generates a MatMul + Relu ONNX model."""
    # Nodes: [M, K] @ [K, N] -> [M, N]
    node_matmul = helper.make_node("MatMul", ["A", "B"], ["C"])
    node_relu = helper.make_node("Relu", ["C"], ["D"])

    # Inputs and Outputs
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 8])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [2, 8])

    graph = helper.make_graph([node_matmul, node_relu], "test-model", [A, B], [D])
    model = helper.make_model(graph, producer_name="seecpp-test")
    
    with open(path, "wb") as f:
        f.write(model.SerializeToString())

def test_full_pipeline():
    onnx_path = "model.onnx"
    sir_path = "model.sir"
    
    # 1. Setup
    create_simple_model(onnx_path)
    
    # 2. Run the SeeC++ Frontend Driver
    # We use check=True to raise an exception if the C++ binary crashes
    result = subprocess.run([FRONTEND_BINARY, onnx_path, sir_path], 
                            capture_output=True, text=True)
    
    print(result.stdout) # Useful for debugging if test fails
    
    assert result.returncode == 0
    assert os.path.exists(sir_path)
    
    # 3. Structural Validation (Requires generated python proto classes)
    # from seecpp_sir_pb2 import BlockProto
    # sir_data = BlockProto()
    # with open(sir_path, "rb") as f:
    #     sir_data.ParseFromString(f.read())
    # assert len(sir_data.operations) == 4 # Input A, Input B, MatMul, Relu
    
    # Cleanup
    if os.path.exists(onnx_path): os.remove(onnx_path)
    if os.path.exists(sir_path): os.remove(sir_path)

if __name__ == "__main__":
    test_full_pipeline()