import torch
import os
from BackendBench.backends import LLMBackend
from BackendBench.llm_client import LLMKernelGenerator
from BackendBench.suite import OpInfoTestSuite

class TestRandomOps:
    suite = OpInfoTestSuite(
        "random_ops_test",
        "cuda",
        torch.float32,
        filter=["bernoulli"]
    )

    def test_bernoulli(self):
        backend = LLMBackend(
            model="mock_model",
            llm_client=LLMKernelGenerator(model="mock_model"),
        )
        backend.generate_kernels(self.suite, attempts=3)

        summary_file = f"{backend.kernels_dir}/bernoulli/bernoulli_summary.txt"
        assert os.path.exists(summary_file)

        with open(summary_file, "r") as f:
            summary = f.read()
            assert "Final Status: ✓ Success" in summary