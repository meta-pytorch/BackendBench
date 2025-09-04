import torch
from BackendBench.suite.base import Test


def get_correctness_tests():
    return [
        Test(lambda: torch.ones(8, device="cuda")),
        (lambda: torch.randn(4, 4, device="cuda"),),
        ((lambda: torch.randn(16, device="cuda")), {"alpha": lambda: 3.0}),
    ]


def get_performance_tests():
    return [
        (lambda: torch.randn(1024, 1024, device="cuda"),),
        (lambda: torch.randn(1_000_000, device="cuda"),),
    ]


