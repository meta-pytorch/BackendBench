import torch
from BackendBench.suite.base import Test


def get_correctness_tests():
    return [
        Test(lambda: torch.ones(8)),
        (lambda: torch.randn(4, 4),),
        ((lambda: torch.randn(16)), {"alpha": lambda: 3.0}),
    ]


def get_performance_tests():
    return [
        (lambda: torch.randn(1024, 1024),),
        (lambda: torch.randn(1_000_000),),
    ]


