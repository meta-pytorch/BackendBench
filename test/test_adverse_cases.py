# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import BackendBench.backends as backends
import BackendBench.multiprocessing_eval as multiprocessing_eval
from BackendBench.suite import TorchBenchOpTest


class TestAdaptiveAvgPool2dBackward:
    @pytest.mark.skip(reason="Skipped due to tensor size causing CUDA OOM in smoke test.")
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
        with multiprocessing_eval.MultiprocessingEvaluator() as evaluator:
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


class TestCase:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class TestMultiprocessingEval:
    def test_multiprocessing_evaluator(self):
        op = torch.relu
        impl = torch.relu  # Same implementation

        correctness_tests = [TestCase([torch.tensor([-1.0, 0.0, 1.0])], {}) for _ in range(3)]
        performance_tests = [TestCase([torch.tensor([-1.0, 0.0, 1.0])], {}) for _ in range(2)]

        with multiprocessing_eval.MultiprocessingEvaluator() as evaluator:
            evaluator.submit_task(op, impl, correctness_tests, performance_tests)

            evaluator.start_evaluation()

            results = evaluator.get_results()

        assert len(results) == 1
        # Should have perfect correctness since using same implementation
        assert results[0].correctness_score == 1.0
        # Performance should be around 1.0 (same speed)
        assert results[0].performance_score.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
