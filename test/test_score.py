# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from BackendBench.score import fastp


def fastp_kernel_bench(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float) -> float:
    """
    Original fastp implementation from kernelBench
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0


class TestFastp:
    def get_results(self, num_tests=100):
        overall_correctness = np.random.randint(0, 2, size=num_tests)
        overall_performance = np.random.uniform(0.5, 2, size=num_tests)
        return overall_correctness, overall_performance


    def test_fastp(self):
        for num_tests in [5, 10, 50, 100]:
            for p in [0, 1, 1.5, 2]:
                overall_correctness, overall_performance = self.get_results(num_tests)

                actual_speed = np.random.randint(1, 101, size=num_tests)
                baseline_speed = actual_speed * overall_performance
                fastp_score_orig = fastp_kernel_bench(overall_correctness, baseline_speed, actual_speed, num_tests, p)

                fastp_score = fastp(overall_correctness.tolist(), overall_performance.tolist(), p)

                assert torch.allclose(fastp_score, torch.tensor(fastp_score_orig, dtype=torch.float32))
