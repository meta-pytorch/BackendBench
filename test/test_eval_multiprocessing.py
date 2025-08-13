import pytest
import torch

try:
    import importlib.util
    from BackendBench.eval_multiprocessing import (
        eval_correctness_multiprocessing,
        eval_one_op_multiprocessing,
    )

    HAS_TRITON = importlib.util.find_spec("triton") is not None
except ImportError:
    HAS_TRITON = False

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton not available")


class TestEvalCorrectnessMultiprocessing:
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

        score = eval_correctness_multiprocessing(op, impl, tests, torch.cuda.device_count())
        assert score == 1.0


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

        correctness, performance = eval_one_op_multiprocessing(
            op, impl, correctness_tests, performance_tests
        )

        # Should have perfect correctness since using same implementation
        assert correctness == 1.0
        # Performance should be around 1.0 (same speed)
        assert performance.item() > 0
