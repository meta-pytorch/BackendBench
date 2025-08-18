import pytest
from BackendBench.torchbench_suite import TorchBenchOpTest
import BackendBench.eval as eval
import BackendBench.backends as backends
import torch


class TestAdaptiveAvgPool2dBackward:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_adaptive_avg_pool2d_backward_gpu(self):
        """Test on GPU with eval_one_op."""
        op_test_should_error = TorchBenchOpTest(
            "aten._adaptive_avg_pool2d_backward.default",
            ["((T([512, 4096, 56, 56], f16), T([512, 4096, 56, 56], f16)), {})"],
            None,
        )

        op_test_should_succeed = TorchBenchOpTest(
            "aten.addmm.default",
            ["((T([14, 14], f32), T([14, 14], f32), T([14, 14], f32)), {})"],
            None,
        )

        # run test that should brick the gpu due to an illegal memory access
        backend = backends.AtenBackend()
        # with pytest.raises(RuntimeError):
        with eval.MultiprocessingEvaluator() as evaluator:
            evaluator.submit_task(
                op_test_should_error.op, 
                backend[op_test_should_error.op], 
                list(op_test_should_error.correctness_tests), 
                list(op_test_should_error.performance_tests),
            )
            evaluator.submit_task(
                op_test_should_succeed.op,
                backend[op_test_should_succeed.op],
                list(op_test_should_succeed.correctness_tests),
                list(op_test_should_succeed.performance_tests),
            )
            evaluator.start_evaluation()
            
            results = evaluator.get_results()

        assert len(results) == 1
        assert results[0].correctness_score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
