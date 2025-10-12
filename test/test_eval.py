# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util

import numpy as np
import pytest
import torch

from BackendBench.eval import (
    allclose,
    cpu_bench,
    eval_correctness,
    eval_correctness_test,
    eval_one_op,
    format_exception,
    perf_at_p,
)

HAS_TRITON = importlib.util.find_spec("triton") is not None

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton not available")


class TestFormatFunctions:
    def test_format_exception(self):
        op = torch.ops.aten.relu.default
        args = [torch.randn(2, 3)]
        kwargs = {"dim": 1}
        exc = ValueError("Test error")

        formatted = format_exception(exc, op, args, kwargs)
        assert "relu.default" in formatted
        assert "(T([2, 3], f32)" in formatted
        assert "dim" in formatted
        assert "Test error" in formatted


class TestAllclose:
    def test_allclose_tensors(self):
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([1.0, 2.0, 3.0])

        assert allclose(tensor1, tensor2) is True

        tensor3 = torch.tensor([1.0, 2.0, 3.01])
        assert allclose(tensor1, tensor3) is True

        tensor_nan1 = torch.tensor([1.0, float("nan"), 3.0])
        tensor_nan2 = torch.tensor([1.0, float("nan"), 3.0])
        assert allclose(tensor_nan1, tensor_nan2) is True

    def test_allclose_scalars(self):
        assert allclose(1, 1) is True
        assert allclose(1.0, 1.0) is True
        assert allclose("test", "test") is True
        assert allclose(1, 2) is False

    def test_allclose_tuples_lists_with_tolerances(self):
        """Test tuple/list comparison with specified tolerances"""
        atol, rtol = 1e-2, 1e-2

        # Lists of tensors - exact match
        list1 = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        list2 = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        assert allclose(list1, list2, atol=atol, rtol=rtol) is True

        # Lists of tensors - within tolerance
        list1 = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        list2 = [torch.tensor([1.01, 2.01]), torch.tensor([3.01, 4.01])]
        assert allclose(list1, list2, atol=atol, rtol=rtol) is True

        # Lists of tensors - outside tolerance
        list1 = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        list2 = [torch.tensor([1.1, 2.1]), torch.tensor([3.1, 4.1])]
        assert allclose(list1, list2, atol=atol, rtol=rtol) is False

        # Tuples of tensors
        tuple1 = (torch.tensor([1.0]), torch.tensor([2.0]))
        tuple2 = (torch.tensor([1.01]), torch.tensor([2.01]))
        assert allclose(tuple1, tuple2, atol=atol, rtol=rtol) is True

        # Nested structures
        nested1 = [[torch.tensor([1.0])], (torch.tensor([2.0]), torch.tensor([3.0]))]
        nested2 = [[torch.tensor([1.01])], (torch.tensor([2.01]), torch.tensor([3.01]))]
        assert allclose(nested1, nested2, atol=atol, rtol=rtol) is True

        # Length mismatch
        list1 = [torch.tensor([1.0]), torch.tensor([2.0])]
        list2 = [torch.tensor([1.0])]
        assert allclose(list1, list2, atol=atol, rtol=rtol) is False


class TestEvalCorrectness:
    def test_eval_correctness_test_pass(self):
        # Use actual torch operations
        op = torch.relu
        impl = torch.relu  # Same implementation should pass

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs

        test = TestCase([torch.tensor([-1.0, 0.0, 1.0])], {})

        result = eval_correctness_test(op, impl, test)
        assert result.has_correct_output is True

    def test_eval_correctness_test_fail(self):
        # Use different operations that produce different results
        op = torch.relu

        def impl(x):
            return x * 2  # Different implementation

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs

        test = TestCase([torch.tensor([1.0, 2.0, 3.0])], {})

        result = eval_correctness_test(op, impl, test)
        assert result.has_correct_output is False

    def test_eval_correctness_test_exception(self):
        op = torch.relu

        def impl_with_error(x):
            raise RuntimeError("Test error")

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs

        test = TestCase([torch.tensor([1.0])], {})

        # Just test that it returns False on exception
        result = eval_correctness_test(op, impl_with_error, test)
        assert result.has_correct_output is False
        assert result.error_msg is not None  # Should have an error message

    def test_eval_correctness_multiple_tests(self):
        op = torch.abs
        impl = torch.abs  # Same implementation

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs

        tests = []
        for i in range(5):
            test = TestCase([torch.tensor([float(i) - 2.5])], {})
            tests.append(test)

        score, correctness_results = eval_correctness(op, impl, tests)
        assert score == 1.0
        assert len(correctness_results) == len(tests)


class TestEvalPerformance:
    def test_cpu_bench(self):
        counter = 0

        def test_fn():
            nonlocal counter
            counter += 1

        # Actually run the benchmark
        time_per_run = cpu_bench(test_fn, num_runs=10)

        # Should have run 10 warmup runs + 10 actual runs = 20 total
        assert counter == 20
        assert time_per_run > 0


class TestEvalOneOp:
    def test_eval_one_op(self):
        op = torch.relu
        impl = torch.relu  # Same implementation

        class TestCase:
            def __init__(self, args, kwargs):
                self.args = args
                self.kwargs = kwargs

        correctness_tests = [TestCase([torch.tensor([-1.0, 0.0, 1.0])], {}) for _ in range(3)]
        performance_tests = [TestCase([torch.tensor([-1.0, 0.0, 1.0])], {}) for _ in range(2)]

        correctness, performance, correctness_results, performance_results = eval_one_op(
            op, impl, correctness_tests, performance_tests
        )

        # Should have perfect correctness since using same implementation
        assert correctness == 1.0
        # Performance should be around 1.0 (same speed)
        assert performance.item() > 0
        # Verbose data should be populated
        assert len(correctness_results) == len(correctness_tests)
        assert len(performance_results) == len(performance_tests)


def fastp_kernel_bench(
    is_correct: np.ndarray,
    baseline_speed: np.ndarray,
    actual_speed: np.ndarray,
    n: int,
    p: float,
) -> float:
    """
    Original fastp implementation from kernelBench
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0


class TestPerfAtP:
    def get_results(self, num_tests=100):
        overall_correctness = np.random.randint(0, 2, size=num_tests)
        overall_performance = np.random.uniform(0.5, 2, size=num_tests)
        return overall_correctness, overall_performance

    def test_perf_at_p(self):
        for num_tests in [5, 10, 50, 100]:
            for p in [0, 1, 1.5, 2]:
                overall_correctness, overall_performance = self.get_results(num_tests)

                actual_speed = np.random.randint(1, 101, size=num_tests)
                baseline_speed = actual_speed * overall_performance
                fastp_score_orig = fastp_kernel_bench(
                    overall_correctness, baseline_speed, actual_speed, num_tests, p
                )

                # Note: The perf@p score calculation here differs subtly from the original fastp score in
                # kernel bench. The original fastp score filters correct samples first, then averages.
                # Here, perf@p averages first, then filters correct samples. Despite this difference,
                # both methods produce equivalent results, so the test remains valid.
                perf_at_p_score = perf_at_p(
                    overall_correctness.tolist(), overall_performance.tolist(), p
                )

                assert torch.allclose(
                    perf_at_p_score, torch.tensor(fastp_score_orig, dtype=torch.float32)
                )
