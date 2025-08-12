import pytest
import torch
from BackendBench.suite import randn, Test, OpTest, TestSuite, SmokeTestSuite
from BackendBench.opregistry import get_operator


class TestRandnFunction:
    def test_randn_returns_callable(self):
        fn = randn(2, 3)
        assert callable(fn)

        tensor = fn()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 3)

    def test_randn_with_kwargs(self):
        fn = randn(2, 3, device="cpu", dtype=torch.float32)
        tensor = fn()

        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32
        assert tensor.shape == (2, 3)


class TestTestClass:
    def test_test_initialization(self):
        test = Test(1, 2, 3, key1="value1", key2="value2")

        assert test._args == (1, 2, 3)
        assert test._kwargs == {"key1": "value1", "key2": "value2"}

    @pytest.mark.skip(reason="Test expects mixed callable/non-callable args - needs clarification")
    def test_test_args_property(self):
        def fn1():
            return 10

        def fn2():
            return 20

        test = Test(fn1, 5, fn2)

        args = test.args
        assert args == [10, 5, 20]

    @pytest.mark.skip(
        reason="Test expects mixed callable/non-callable kwargs - needs clarification"
    )
    def test_test_kwargs_property(self):
        def fn1():
            return "computed"

        test = Test(key1=fn1, key2="static")

        kwargs = test.kwargs
        assert kwargs == {"key1": "computed", "key2": "static"}

    def test_test_with_randn(self):
        test = Test(randn(2, 3), randn(3, 4), device=lambda: "cpu")

        args = test.args
        kwargs = test.kwargs

        assert len(args) == 2
        assert isinstance(args[0], torch.Tensor)
        assert isinstance(args[1], torch.Tensor)
        assert args[0].shape == (2, 3)
        assert args[1].shape == (3, 4)
        assert kwargs == {"device": "cpu"}


class TestOpTest:
    def test_optest_initialization(self):
        op = torch.ops.aten.relu.default
        correctness_tests = [Test(randn(2, 2))]
        performance_tests = [Test(randn(100, 100))]

        optest = OpTest(op, correctness_tests, performance_tests)

        assert optest.op == op
        assert optest.correctness_tests == correctness_tests
        assert optest.performance_tests == performance_tests

    def test_optest_attributes(self):
        op = torch.ops.aten.add.Tensor
        correctness_tests = [Test(randn(2, 2), randn(2, 2)), Test(randn(3, 3), randn(3, 3))]
        performance_tests = [Test(randn(1000, 1000), randn(1000, 1000))]

        optest = OpTest(op, correctness_tests, performance_tests)

        assert len(optest.correctness_tests) == 2
        assert len(optest.performance_tests) == 1


class TestTestSuite:
    def test_testsuite_initialization(self):
        optests = [
            OpTest(torch.ops.aten.relu.default, [Test(randn(2))], [Test(randn(100))]),
            OpTest(
                torch.ops.aten.add.Tensor,
                [Test(randn(2), randn(2))],
                [Test(randn(100), randn(100))],
            ),
        ]

        suite = TestSuite("test_suite", optests)

        assert suite.name == "test_suite"
        assert suite.optests == optests

    def test_testsuite_iteration(self):
        optests = [
            OpTest(torch.ops.aten.relu.default, [Test(randn(2))], [Test(randn(100))]),
            OpTest(
                torch.ops.aten.add.Tensor,
                [Test(randn(2), randn(2))],
                [Test(randn(100), randn(100))],
            ),
        ]

        suite = TestSuite("test_suite", optests)

        collected = list(suite)
        assert len(collected) == 2
        assert collected[0].op == torch.ops.aten.relu.default
        assert collected[1].op == torch.ops.aten.add.Tensor


class TestSmokeTestSuiteStructure:
    def test_smoke_test_suite_exists(self):
        assert isinstance(SmokeTestSuite, TestSuite)
        assert SmokeTestSuite.name == "smoke"

    def test_smoke_test_suite_contains_relu(self):
        optests = list(SmokeTestSuite)

        assert len(optests) >= 1
        assert optests[0].op == get_operator(torch.ops.aten.relu.default)

        # Check correctness tests
        assert len(optests[0].correctness_tests) >= 1
        correctness_test = optests[0].correctness_tests[0]
        args = correctness_test.args
        assert len(args) == 1
        assert isinstance(args[0], torch.Tensor)

        # Check performance tests
        assert len(optests[0].performance_tests) >= 1
        perf_test = optests[0].performance_tests[0]
        perf_args = perf_test.args
        assert len(perf_args) == 1
        assert isinstance(perf_args[0], torch.Tensor)
        assert perf_args[0].numel() > args[0].numel()


class TestSuiteIntegration:
    def test_suite_with_multiple_operations(self):
        optests = [
            OpTest(
                torch.ops.aten.relu.default,
                [Test(randn(2, 2)), Test(randn(3, 3))],
                [Test(randn(100, 100))],
            ),
            OpTest(torch.ops.aten.sigmoid.default, [Test(randn(4, 4))], [Test(randn(200, 200))]),
            OpTest(
                torch.ops.aten.add.Tensor,
                [Test(randn(2, 2), randn(2, 2))],
                [Test(randn(100, 100), randn(100, 100))],
            ),
        ]

        suite = TestSuite("integration_test", optests)

        ops_found = [optest.op for optest in suite]
        assert torch.ops.aten.relu.default in ops_found
        assert torch.ops.aten.sigmoid.default in ops_found
        assert torch.ops.aten.add.Tensor in ops_found

    def test_test_args_evaluation_timing(self):
        counter = 0

        def counting_fn():
            nonlocal counter
            counter += 1
            return counter

        test = Test(counting_fn)

        assert test.args == [1]
        assert test.args == [2]
        assert test.args == [3]
