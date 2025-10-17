import torch
import os
from BackendBench.backends import LLMBackend
from BackendBench.llm_client import LLMKernelGenerator
from BackendBench.suite import OpInfoTestSuite


class TestCase:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs
class TestTensorCreationOps:
    suite = OpInfoTestSuite(
        "tensor_creation_ops_test",
        "cuda",
        torch.float32,
        filter=["new_empty", "new_empty_strided", "new_full", "new_ones", "new_zeros"],
    )

    def test_tensor_creation_ops(self):
        backend = LLMBackend(
            model="mock_model",
            llm_client=LLMKernelGenerator(model="mock_model"),
        )
        backend.generate_kernels(self.suite, attempts=3)

        for op_name in self.suite.get_op_names():
            summary_file = f"{backend.kernels_dir}/{op_name}/{op_name}_summary.txt"
            assert os.path.exists(summary_file)

            with open(summary_file, "r") as f:
                summary = f.read()
                assert "Final Status: ✓ Success" in summary

    def test_new_empty(self):
        base_tensor = torch.ones((2, 3), device="cuda", dtype=torch.float32)
        new_tensor = base_tensor.new_empty((4, 5))

        assert new_tensor.shape == (4, 5)
        assert new_tensor.device == base_tensor.device
        assert new_tensor.dtype == base_tensor.dtype
        assert new_tensor.is_contiguous()
        assert new_tensor.numel() > 0

    def test_new_empty_strided(self):
        base_tensor = torch.ones((2, 3), device="cuda", dtype=torch.float32)
        new_tensor = base_tensor.new_empty_strided((4, 5), (10, 2))

        assert new_tensor.shape == (4, 5)
        assert new_tensor.stride() == (10, 2)
        assert new_tensor.device == base_tensor.device
        assert new_tensor.dtype == base_tensor.dtype
        assert new_tensor.numel() > 0

    def test_new_full(self):
        base_tensor = torch.ones((2, 3), device="cuda", dtype=torch.float32)
        fill_value = 7.0
        new_tensor = base_tensor.new_full((4, 5), fill_value)

        assert new_tensor.shape == (4, 5)
        assert new_tensor.device == base_tensor.device
        assert new_tensor.dtype == base_tensor.dtype
        assert torch.all(new_tensor == fill_value)

    def test_new_ones(self):
        base_tensor = torch.ones((2, 3), device="cuda", dtype=torch.float32)
        new_tensor = base_tensor.new_ones((4, 5))

        assert new_tensor.shape == (4, 5)
        assert new_tensor.device == base_tensor.device
        assert new_tensor.dtype == base_tensor.dtype
        assert torch.all(new_tensor == 1.0)

    def test_new_zeros(self):
        base_tensor = torch.ones((2, 3), device="cuda", dtype=torch.float32)
        new_tensor = base_tensor.new_zeros((4, 5))

        assert new_tensor.shape == (4, 5)
        assert new_tensor.device == base_tensor.device
        assert new_tensor.dtype == base_tensor.dtype
        assert torch.all(new_tensor == 0.0)