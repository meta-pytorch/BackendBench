import torch


def randn(*args, **kwargs):
    return lambda: torch.randn(*args, **kwargs)


class Test:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self):
        return [arg() for arg in self._args]

    @property
    def kwargs(self):
        return {k: v() for k, v in self._kwargs.items()}


class OpTest:
    def __init__(self, op, correctness_tests, performance_tests):
        self.op = op
        self.correctness_tests = correctness_tests
        self.performance_tests = performance_tests


class TestSuite:
    def __init__(self, name, optests):
        self.name = name
        self.optests = optests

    def __iter__(self):
        for optest in self.optests:
            yield optest


SmokeTestSuite = TestSuite(
    "smoke",
    [
        OpTest(
            torch.ops.aten.relu.default,
            [
                Test(randn(2, device="cuda")),
            ],
            [
                Test(randn(2**28, device="cuda")),
            ],
        )
    ],
)
