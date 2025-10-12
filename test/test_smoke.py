# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

import BackendBench.backends as backends
from BackendBench.eval import eval_one_op
from BackendBench.suite import SmokeTestSuite


class TestSmoke:
    @pytest.fixture
    def aten_backend(self):
        return backends.AtenBackend()

    def test_smoke_suite_aten_backend(self, aten_backend):
        overall_correctness = []
        overall_performance = []

        for test in SmokeTestSuite:
            if test.op not in aten_backend:
                pytest.skip(f"Operation {test.op} not in backend")

            correctness, perf, correctness_results, performance_results = eval_one_op(
                test.op,
                aten_backend[test.op],
                test.correctness_tests,
                test.performance_tests,
            )

            is_correct = all(result.has_correct_output for result in correctness_results)
            overall_correctness.append(is_correct)
            overall_performance.append(perf)

            assert len(correctness_results) == len(test.correctness_tests)
            assert len(performance_results) == len(test.performance_tests)

            assert correctness > 0, f"Operation {test.op} failed all correctness tests"
            assert perf > 0.1, f"Operation {test.op} is more than 10x slower than reference"

        mean_correctness = torch.tensor(overall_correctness).float().mean().item()
        geomean_perf = torch.tensor(overall_performance).log().mean().exp().item()

        assert mean_correctness >= 0.8, (
            f"Mean correctness {mean_correctness:.2f} is below threshold of 0.8"
        )
        assert geomean_perf >= 0.5, (
            f"Geomean performance {geomean_perf:.2f} is below threshold of 0.5"
        )

        print(f"Correctness score (mean pass rate): {mean_correctness:.2f}")
        print(f"Performance score (geomean speedup): {geomean_perf:.2f}")
