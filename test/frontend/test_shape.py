@pytest.mark.parametrize("m, k, n", [(2, 4, 8), (1, 128, 64)])
def test_shape_inference_matmul(runner, tmp_path, m, k, n):
    """Verify M*K @ K*N produces M*N results."""
    onnx_file = tmp_path / "matmul.onnx"
    sir_file = tmp_path / "matmul.sir"

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [m, n])
    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    
    graph = helper.make_graph([node], "mm", [A, B], [C])
    with open(onnx_file, "wb") as f:
        f.write(helper.make_model(graph).SerializeToString())

    sir_block = runner.run_and_parse(str(onnx_file), str(sir_file))
    
    mm_op = [op for op in sir_block.operations if "MatMul" in op.mnemonic][0]
    result_shape = list(mm_op.results[0].shape.dims)
    assert result_shape == [m, n]