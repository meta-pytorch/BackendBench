import logging

import torch

try:
    import triton.testing
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


from BackendBench.utils import uses_cuda_stream
from BackendBench.utils import serialize_args

logger = logging.getLogger(__name__)

EXC_MSG = """
Exception raised for {op}:
    args: {args}
    exc: {exc}
"""


def format_exception(e, op, args, kwargs):
    op_name = getattr(op, "__name__", str(op))
    return EXC_MSG.format(op=op_name, args=serialize_args(args, kwargs), exc=e)


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
        logging.debug(f"Testing {op.__name__} with args {serialize_args(test.args, test.kwargs)}")
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
    bench_fn = (triton.testing.do_bench if TRITON_AVAILABLE and torch.cuda.is_available() else cpu_bench)
    base_times = []
    test_times = []
    for test in tests:
        logging.debug(
            f"Benchmarking {op.__name__} with args {serialize_args(test.args, test.kwargs)}"
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
    # TODO: We should have proper error reporting instead of just saying this is 0,
    # but that should be a separate PR.
    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        return 0, 0
    return eval_correctness(op, impl, correctness_tests), eval_performance(
        op, impl, performance_tests
    )
