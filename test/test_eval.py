import pytest
import torch

try:
    import importlib.util
    from BackendBench.eval import (
        format_tensor,
        format_args,
        format_kwargs,
        format_exception,
        allclose,
        eval_correctness_test,
        eval_correctness,
        eval_performance,
        eval_one_op,
        cpu_bench,
    )

    HAS_TRITON = importlib.util.find_spec("triton") is not None
except ImportError:
    HAS_TRITON = False

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton not available")


class TestFormatFunctions:
    def test_format_tensor(self):
        tensor = torch.randn(2, 3, 4, dtype=torch.float32)
        formatted = format_tensor(tensor)
        assert formatted == "torch.float32[2, 3, 4]"

        tensor_int = torch.randint(0, 10, (5, 5), dtype=torch.int64)
        formatted_int = format_tensor(tensor_int)
        assert formatted_int == "torch.int64[5, 5]"

    def test_format_args(self):
        tensor1 = torch.randn(2, 3)
        tensor2 = torch.randn(3, 4)
        scalar = 2.5

        args = [tensor1, scalar, tensor2]
        formatted = format_args(args)

        assert len(formatted) == 3
        assert formatted[0] == "torch.float32[2, 3]"
        assert formatted[1] == 2.5
        assert formatted[2] == "torch.float32[3, 4]"

    def test_format_kwargs(self):
        tensor = torch.randn(2, 3)
        kwargs = {"input": tensor, "dim": 1, "keepdim": True}

        formatted = format_kwargs(kwargs)
        assert formatted["input"] == "torch.float32[2, 3]"
        assert formatted["dim"] == 1
        assert formatted["keepdim"] is True

    def test_format_exception(self):
        op = torch.ops.aten.relu.default
        args = [torch.randn(2, 3)]
        kwargs = {"dim": 1}
        exc = ValueError("Test error")

        formatted = format_exception(exc, op, args, kwargs)
        assert "aten.relu.default" in formatted
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
        impl = lambda x: x * 2  # Different implementation
        
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
        
        # Should have run warmup (10%) + actual runs
        assert counter == 11  # 1 warmup + 10 runs
        assert time_per_run > 0

    @pytest.mark.skip(reason="Performance testing requires controlled environment")
    def test_eval_performance_cpu(self):
        # Performance tests are not reliable in unit tests
        pass

    @pytest.mark.skip(reason="Performance testing requires controlled environment")
    def test_eval_performance_cuda(self):
        # Performance tests are not reliable in unit tests
        pass


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
        
        correctness, performance = eval_one_op(
            op, impl, correctness_tests, performance_tests
        )
        
        # Should have perfect correctness since using same implementation
        assert correctness == 1.0
        # Performance should be around 1.0 (same speed)
        assert performance.item() > 0
