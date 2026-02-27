import numpy as np

def test_constant_ingestion(runner, tmp_path):
    """Test if ONNX initializers become SIR constants with correct shapes."""
    onnx_file = tmp_path / "const.onnx"
    sir_file = tmp_path / "const.sir"

    # Create an initializer (weight)
    weight_data = np.random.randn(3, 3).astype(np.float32)
    weight_tensor = helper.make_tensor("W", TensorProto.FLOAT, [3, 3], weight_data.flatten())
    
    # Add a dummy node that uses the weight
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])
    node = helper.make_node("Add", ["X", "W"], ["Y"])

    graph = helper.make_graph([node], "const_test", [X], [Y], initializer=[weight_tensor])
    with open(onnx_file, "wb") as f:
        f.write(helper.make_model(graph).SerializeToString())

    sir_block = runner.run_and_parse(str(onnx_file), str(sir_file))
    
    # Rigor check: There should be a constant op for 'W'
    const_ops = [op for op in sir_block.operations if "constant" in op.mnemonic]
    assert any(op.results[0].id == "W" for op in const_ops)
    
    w_op = [op for op in const_ops if op.results[0].id == "W"][0]
    assert list(w_op.results[0].shape.dims) == [3, 3]