import onnx
from onnx import helper, TensorProto

def test_ingressor_identity_map(runner, tmp_path):
    """Test if Ingressor correctly preserves node connectivity."""
    onnx_file = tmp_path / "identity.onnx"
    sir_file = tmp_path / "identity.sir"

    # Create: Input -> Relu -> Output
    node = helper.make_node("Relu", ["X"], ["Y"])
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph)
    
    with open(onnx_file, "wb") as f:
        f.write(model.SerializeToString())

    sir_block = runner.run_and_parse(str(onnx_file), str(sir_file))
    
    # Validation: Look for the Relu op and verify its operand is linked to the Input op
    relu_ops = [op for op in sir_block.operations if "relu" in op.mnemonic]
    assert len(relu_ops) == 1
    assert relu_ops[0].operand_ids[0] == "X"