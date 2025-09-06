import torch
from BackendBench.suite.base import Test


def get_correctness_tests():
    return [
        Test(lambda: torch.ones(8, device="cuda")),
    ]


def get_performance_tests():
    return [
        Test(lambda: torch.ones(8, device="cuda")),
    ]
