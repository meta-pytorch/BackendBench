import torch
import os
from BackendBench.backends import LLMBackend
from BackendBench.llm_client import LLMKernelGenerator
from BackendBench.suite import OpInfoTestSuite

class TestTensorCreationOps:
    suite = OpInfoTestSuite(
        "tensor_creation_ops_test",
        "cuda",
        torch.float32,
        filter=["cat", "clone", "copy_", "elu_backward", "masked_fill_", "new_empty", "new_empty_strided", "new_full", "new_ones", "new_zeros", "nonzero", "repeat", "split", "split_with_sizes", "unsqueeze_"],
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