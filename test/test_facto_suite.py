# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import BackendBench.backends as backends
from BackendBench.eval import eval_one_op
from BackendBench.facto_suite import FactoTestSuite

import importlib.util

HAS_FACTO_DEPS = importlib.util.find_spec("facto") is not None

pytestmark = pytest.mark.skipif(not HAS_FACTO_DEPS, reason="facto dependencies not available")


class TestFactoSuite:
    def test_facto_suite_relu_default_correctness_not_empty(self):
        ops = ["relu.default"]
        num_runs = 10
        empty = False
        probability = 1.0

        suite = FactoTestSuite(
            name="facto_relu_test",
            device="cuda",
            dtype=torch.bfloat16,
            filter=ops,
            num_runs=num_runs,
            empty=empty,
            probability=probability,
        )

        backend = backends.AtenBackend()

        # Track overall correctness and performance
        overall_correctness = []

        # Iterate through the test suite (should contain relu operations)
        for test in suite:
            for ctest in test.correctness_tests:
                for arg in ctest.args:
                    if isinstance(arg, torch.Tensor):
                        # assert args not empty
                        assert arg.numel() > 0, f"Tensor arg is empty for {test.op}"
                for key, value in ctest.kwargs.items():
                    if isinstance(value, torch.Tensor):
                        # assert kwargs not empty
                        assert value.numel() > 0, f"Tensor kwarg is empty for {test.op}"

            # Evaluate the operation
            correctness, _ = eval_one_op(
                test.op,
                backend[test.op],  # AtenBackend returns the original op
                test.correctness_tests,
                test.performance_tests,
            )
            print(f"Correctness for {test.op}: {correctness}")
            overall_correctness.append(correctness)

            # Individual test assertions
            assert correctness > 0, f"Operation {test.op} failed all correctness tests"

        # Calculate mean correctness
        mean_correctness = torch.tensor(overall_correctness).mean().item()

        # Main assertion: correctness should be > 0.8
        assert mean_correctness > 0.8, (
            f"Mean correctness {mean_correctness:.2f} is not > 0.8 for relu.default"
        )

    def test_facto_suite_num_run(self):
        ops = ["relu.default"]
        num_runs = 10
        empty = False
        probability = 1.0

        suite = FactoTestSuite(
            name="facto_relu_test",
            device="cuda",
            dtype=torch.bfloat16,
            filter=ops,
            num_runs=num_runs,
            empty=empty,
            probability=probability,
        )

        for test in suite:
            assert len(list(test.correctness_tests)) == num_runs, (
                f"Number of correctness tests for {test.op} is not {num_runs}"
            )
