import pytest
import torch
from unittest.mock import Mock, patch

try:
    import triton.testing
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

    HAS_TRITON = True
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
        op = Mock(__name__="test_op")
        args = [torch.randn(2, 3)]
        kwargs = {"dim": 1}
        exc = ValueError("Test error")

        formatted = format_exception(exc, op, args, kwargs)
        assert "test_op" in formatted
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
        op = Mock(return_value=torch.tensor([2.0]))
        op.__name__ = "add_one"

        impl = Mock(return_value=torch.tensor([2.0]))

        test = Mock()
        test.args = [torch.tensor([1.0])]
        test.kwargs = {}

        result = eval_correctness_test(op, impl, test)
        assert result is True

    def test_eval_correctness_test_fail(self):
        op = Mock(return_value=torch.tensor([2.0]))
        op.__name__ = "add_one"

        impl = Mock(return_value=torch.tensor([3.0]))

        test = Mock()
        test.args = [torch.tensor([1.0])]
        test.kwargs = {}

        result = eval_correctness_test(op, impl, test)
        assert result is False

    def test_eval_correctness_test_exception(self):
        op = Mock(return_value=torch.tensor([2.0]))
        op.__name__ = "test_op"

        impl = Mock(side_effect=RuntimeError("Test error"))

        test = Mock()
        test.args = [torch.tensor([1.0])]
        test.kwargs = {}

        with patch("BackendBench.eval.logger") as mock_logger:
            result = eval_correctness_test(op, impl, test)
            assert result is False
            mock_logger.warning.assert_called_once()

    def test_eval_correctness_multiple_tests(self):
        op = Mock(return_value=torch.tensor([2.0]))
        op.__name__ = "test_op"

        impl = Mock(return_value=torch.tensor([2.0]))

        tests = []
        for i in range(5):
            test = Mock()
            test.args = [torch.tensor([float(i)])]
            test.kwargs = {}
            tests.append(test)

        with patch("BackendBench.eval.logging"):
            score = eval_correctness(op, impl, tests)
            assert score == 1.0


class TestEvalPerformance:
    def test_cpu_bench(self):
        counter = 0

        def test_fn():
            nonlocal counter
            counter += 1

        with patch("time.perf_counter", side_effect=[0.0, 0.0, 1.0]):
            time_per_run = cpu_bench(test_fn, num_runs=100)

        assert counter == 110
        assert time_per_run == 0.01

    @patch("torch.cuda.is_available", return_value=False)
    def test_eval_performance_cpu(self, mock_cuda):
        op = Mock(side_effect=lambda x: x + 1)
        op.__name__ = "test_op"

        impl = Mock(side_effect=lambda x: x + 1)

        tests = []
        for i in range(3):
            test = Mock()
            test.args = [torch.tensor([float(i)])]
            test.kwargs = {}
            tests.append(test)

        with patch("BackendBench.eval.cpu_bench") as mock_bench:
            mock_bench.side_effect = [0.002, 0.001] * len(tests)

            speedup = eval_performance(op, impl, tests)

            assert abs(speedup.item() - 2.0) < 0.01

    @patch("torch.cuda.is_available", return_value=True)
    @patch("triton.testing.do_bench")
    def test_eval_performance_cuda(self, mock_do_bench, mock_cuda):
        op = Mock(side_effect=lambda x: x + 1)
        op.__name__ = "test_op"

        impl = Mock(side_effect=lambda x: x + 1)

        test = Mock()
        test.args = [torch.tensor([1.0])]
        test.kwargs = {}

        mock_do_bench.side_effect = [0.002, 0.001]

        speedup = eval_performance(op, impl, [test])
        assert abs(speedup.item() - 2.0) < 0.01


class TestEvalOneOp:
    def test_eval_one_op(self):
        op = Mock(return_value=torch.tensor([2.0]))
        op.__name__ = "test_op"

        impl = Mock(return_value=torch.tensor([2.0]))

        correctness_tests = [Mock(args=[torch.tensor([1.0])], kwargs={}) for _ in range(3)]
        performance_tests = [Mock(args=[torch.tensor([1.0])], kwargs={}) for _ in range(2)]

        with patch("BackendBench.eval.eval_correctness", return_value=0.9) as mock_correct:
            with patch(
                "BackendBench.eval.eval_performance", return_value=torch.tensor(1.5)
            ) as mock_perf:
                correctness, performance = eval_one_op(
                    op, impl, correctness_tests, performance_tests
                )

        assert correctness == 0.9
        assert performance.item() == 1.5

        mock_correct.assert_called_once_with(op, impl, correctness_tests)
        mock_perf.assert_called_once_with(op, impl, performance_tests)
