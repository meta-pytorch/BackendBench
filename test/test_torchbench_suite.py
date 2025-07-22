import pytest
import torch
from BackendBench.torchbench_suite import TorchBenchOpTest


class TestOpTest:
    def test_op_test(self):
        op_test = TorchBenchOpTest("aten.relu.default", ["((T([32, 128, 512], f16),), {})"], None)
        for test in op_test.correctness_tests:
            args, kwargs = test.args, test.kwargs
            arg, *extras = args
            assert arg.shape == torch.Size([32, 128, 512])
            assert arg.dtype == torch.float16
            assert kwargs == {}
            assert extras == []

            torch.testing.assert_close(torch.relu(arg), op_test.op(arg))

    def test_topn(self):
        op_test = TorchBenchOpTest(
            "aten.relu.default",
            [
                "((T([32, 128, 512], f16),), {})",
                "((T([32, 256, 512], f16),), {})",
            ],
            1,
        )
        assert len(op_test.tests()) == 1
        for test in op_test.correctness_tests:
            args, kwargs = test.args, test.kwargs
            arg, *extras = args
            assert arg.shape == torch.Size([32, 256, 512])
            assert arg.dtype == torch.float16
            assert kwargs == {}
            assert extras == []

            torch.testing.assert_close(torch.relu(arg), op_test.op(arg))
