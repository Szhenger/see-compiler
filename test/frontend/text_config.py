import subprocess
import os
import sir_pb2  # Generated from sir.proto

class FrontendRunner:
    def __init__(self, binary_path):
        self.binary_path = binary_path

    def run_and_parse(self, onnx_path, sir_path):
        result = subprocess.run([self.binary_path, onnx_path, sir_path], 
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Frontend failed: {result.stderr}")
        
        block = sir_pb2.BlockProto()
        with open(sir_path, "rb") as f:
            block.ParseFromString(f.read())
        return block

@pytest.fixture
def runner():
    return FrontendRunner("../build/source/tools/frontend_driver")