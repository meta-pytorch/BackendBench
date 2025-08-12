import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import torch

import triton.testing


from BackendBench.utils import uses_cuda_stream, serialize_args, deserialize_args
from BackendBench.opregistry import get_operator, _extract_spec_name_from_op

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


def _run_single_test(op_spec_name, impl, args, kwargs):
    """Helper function to run a single test in a subprocess."""
    # args, kwargs = (inps)
    op = get_operator(op_spec_name)
    try:
        ref = op(*args, **kwargs)
        res = impl(*args, **kwargs)
        return allclose(ref, res)
    except Exception as e:
        # Note: logging in subprocess may not work as expected
        return False


def eval_correctness_multiprocessing(op, impl, tests, num_workers=10):
    """
    Multiprocessing version of eval_correctness_test.
    Runs the test in a separate process to isolate CUDA errors.
    """
    correct, total = 0, 0
    multiprocessing.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _run_single_test, _extract_spec_name_from_op(op), impl, test.args, test.kwargs
            ): None
            for test in tests
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                correct += 1
            total += 1
    return correct / total


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
    bench_fn = triton.testing.do_bench if torch.cuda.is_available() else cpu_bench
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


def eval_one_op(op, impl, correctness_tests, performance_tests, num_workers=0):
    """Evaluate impl of op against correctness_tests and performance_tests."""
    # TODO: We should have proper error reporting instead of just saying this is 0,
    # but that should be a separate PR.
    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        return 0, 0
    if num_workers > 0:
        eval_correctness_fun = lambda op, impl, correctness_tests: eval_correctness_multiprocessing(
            op, impl, correctness_tests, num_workers
        )
    else:
        eval_correctness_fun = eval_correctness
    return eval_correctness_fun(op, impl, correctness_tests), eval_performance(
        op, impl, performance_tests
    )
