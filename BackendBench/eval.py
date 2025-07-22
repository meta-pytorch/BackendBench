import logging

import torch

import triton.testing


logger = logging.getLogger(__name__)

EXC_MSG = """
Exception raised for {op}:
    args: {args}
    kwargs: {kwargs}
    exc: {exc}
"""


def format_tensor(t):
    return f"{t.dtype}{list(t.shape)}"


def format_args(args):
    return [format_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]


def format_kwargs(kwargs):
    return {k: format_tensor(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}


def format_exception(e, op, args, kwargs):
    op_name = getattr(op, "__name__", str(op))
    return EXC_MSG.format(op=op_name, args=format_args(args), kwargs=format_kwargs(kwargs), exc=e)


def allclose(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, equal_nan=True, atol=1e-2, rtol=1e-2)
        return True
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
        return all(allclose(x, y) for x, y in zip(a, b))
    return a == b


def eval_correctness_test(op, impl, test):
    """Evaluate impl of op against test."""
    args, kwargs = test.args, test.kwargs
    ref = op(*args, **kwargs)
    try:
        res = impl(*args, **kwargs)
        return allclose(ref, res)
    except Exception as e:
        logger.warning(format_exception(e, op, args, kwargs))
        return False


def eval_correctness(op, impl, tests):
    correct, total = 0, 0
    for test in tests:
        logging.debug(
            f"Testing {op.__name__} with args {format_args(test.args)} and kwargs {format_kwargs(test.kwargs)}"
        )
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
    bench_fn = triton.testing.do_bench if torch.cuda.is_available() else cpu_bench
    base_times = []
    test_times = []
    for test in tests:
        logging.debug(
            f"Benchmarking {op.__name__} with args {format_args(test.args)} and kwargs {format_kwargs(test.kwargs)}"
        )
        base_times.append(bench_fn(lambda: op(*test.args, **test.kwargs)))
        try:
            allclose(op(*test.args, **test.kwargs), impl(*test.args, **test.kwargs))
        except Exception:
            test_times.append(base_times[-1])
            continue
        test_times.append(bench_fn(lambda: impl(*test.args, **test.kwargs)))
    speedups = torch.tensor(base_times) / torch.tensor(test_times)
    return speedups.log().mean().exp()


def eval_one_op(op, impl, correctness_tests, performance_tests):
    """Evaluate impl of op against correctness_tests and performance_tests."""
    return eval_correctness(op, impl, correctness_tests), eval_performance(
        op, impl, performance_tests
    )
