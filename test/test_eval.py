# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

try:
    import importlib.util
    from BackendBench.eval import (
        format_exception,
        allclose,
        eval_correctness_test,
        eval_correctness,
        eval_one_op,
        cpu_bench,
        gpu_bench,
    )

    HAS_TRITON = importlib.util.find_spec("triton") is not None
except ImportError:
    HAS_TRITON = False

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton not available")


class TestFormatFunctions:
    def test_format_exception(self):
        op = torch.ops.aten.relu.default
        args = [torch.randn(2, 3)]
        kwargs = {"dim": 1}
        exc = ValueError("Test error")

        formatted = format_exception(exc, op, args, kwargs)
        assert "relu.default" in formatted
        assert "torch.float32[2, 3]" in formatted
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

    def test_allclose_lists(self):
        list1 = [torch.tensor([1.0]), torch.tensor([2.0])]
        list2 = [torch.tensor([1.0]), torch.tensor([2.0])]

        assert allclose(list1, list2) is True

        list3 = [torch.tensor([1.0])]
        with pytest.raises(Exception):
            allclose(list1, list3)

    def test_allclose_scalars(self):
        assert allclose(1, 1) is True
        assert allclose(1.0, 1.0) is True
        assert allclose("test", "test") is True
        assert allclose(1, 2) is False


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
        assert result is True

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
        assert result is False

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
        assert result is False

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

        score = eval_correctness(op, impl, tests)
        assert score == 1.0


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

    def test_gpu_bench(self):
        counter = 0

        def test_fn():
            nonlocal counter
            counter += 1

        time_per_run = gpu_bench(test_fn, num_runs=10)

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

        correctness, performance = eval_one_op(op, impl, correctness_tests, performance_tests)

        # Should have perfect correctness since using same implementation
        assert correctness[0]
        # Performance should be around 1.0 (same speed)
        assert performance.item() > 0
