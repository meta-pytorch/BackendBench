# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import BackendBench.multiprocessing_eval as multiprocessing_eval


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
