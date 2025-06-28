import logging

import torch
from triton.testing import do_bench

logger = logging.getLogger(__name__)


def allclose(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, equal_nan=True)
        return True
    if isinstance(a, (list, tuple)):
        return all(allclose(x, y) for x, y in zip(a, b))
    return a == b


EXC_MSG = """
Exception raised for {op}:
    args: {args}
    kwargs: {kwargs}
    exc: {exc}
"""


def eval_correctness_test(op, impl, test):
    """Evaluate impl of op against test."""
    args, kwargs = test.args, test.kwargs
    ref = op(*args, **kwargs)
    try:
        res = impl(*args, **kwargs)
        return allclose(ref, res)
    except Exception as e:
        logger.debug(EXC_MSG.format(op=op, args=args, kwargs=kwargs, exc=e))
        return False


def eval_correctness(op, impl, tests):
    correct, total = 0, 0
    for test in tests:
        if eval_correctness_test(op, impl, test):
            correct += 1
        total += 1
    return correct / total


def cpu_bench(fn, num_runs=100):
    """Simple CPU benchmarking using time.perf_counter."""
    import time

    for _ in range(10):
        fn()

    start = time.perf_counter()
    for _ in range(num_runs):
        fn()
    return (time.perf_counter() - start) / num_runs


def eval_performance(op, impl, tests):
    if torch.cuda.is_available():
        base_times = [do_bench(lambda: op(*test.args, **test.kwargs)) for test in tests]
        test_times = [do_bench(lambda: impl(*test.args, **test.kwargs)) for test in tests]
    else:
        base_times = [cpu_bench(lambda: op(*test.args, **test.kwargs)) for test in tests]
        test_times = [cpu_bench(lambda: impl(*test.args, **test.kwargs)) for test in tests]

    speedups = torch.tensor(test_times) / torch.tensor(base_times)
    # geometric mean of speedups
    return speedups.log().mean().exp()


def eval_one_op(op, impl, correctness_tests, performance_tests):
    """Evaluate impl of op against correctness_tests and performance_tests."""
    return eval_correctness(op, impl, correctness_tests), eval_performance(
        op, impl, performance_tests
    )
