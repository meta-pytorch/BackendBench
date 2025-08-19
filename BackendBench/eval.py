# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    if torch.cuda.is_available():
        import triton.testing

        TRITON_AVAILABLE = True
    else:
        TRITON_AVAILABLE = False
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


def compute_errors(ref, res) -> Tuple[Optional[float], Optional[float]]:
    """Compute absolute and relative errors between reference and result tensors.

    Returns:
        Tuple of (absolute_error, relative_error) or (None, None) if not tensors/list of tensors
    """
    if isinstance(ref, torch.Tensor) and isinstance(res, torch.Tensor):
        if ref.shape != res.shape:
            return None, None

        # Convert to float for error calculation
        ref_float = ref.float()
        res_float = res.float()

        # Absolute error
        abs_error = (ref_float - res_float).abs().mean().item()

        # Relative error (avoid division by zero)
        ref_abs = ref_float.abs()
        rel_error = ((ref_float - res_float).abs() / (ref_abs + 1e-10)).mean().item()

        return abs_error, rel_error
    elif isinstance(ref, (list, tuple)) and isinstance(res, (list, tuple)):
        if len(ref) != len(res):
            return None, None

        # For lists/tuples, compute mean error across all elements.
        # We will return the mean of these means
        mean_abs_error = 0.0
        mean_rel_error = 0.0

        for r, s in zip(ref, res):
            abs_err, rel_err = compute_errors(r, s)
            mean_abs_error += abs_err
            mean_rel_error += rel_err

        return mean_abs_error / len(ref), mean_rel_error / len(ref)
    else:
        return None, None


def eval_correctness_test(
    op, impl, test
) -> Tuple[bool, Optional[str], Optional[float], Optional[float]]:
    """Evaluate impl of op against test.

    Returns:
        Tuple of (is_correct, error_message, absolute_error, relative_error)
    """
    args, kwargs = test.args, test.kwargs
    ref = op(*args, **kwargs)
    try:
        res = impl(*args, **kwargs)
        is_correct = allclose(ref, res)

        # Compute errors even if test passes (for verbose mode)
        abs_error, rel_error = compute_errors(ref, res)

        return is_correct, None, abs_error, rel_error
    except Exception as e:
        error_msg = format_exception(e, op, args, kwargs)
        logger.warning(error_msg)
        return False, str(e), None, None


def eval_correctness(op, impl, tests, verbose_data: defaultdict):
    """Evaluate correctness of impl against tests."""
    correct, total = 0, 0
    for test in tests:
        args_str = serialize_args(test.args, test.kwargs)
        logging.debug(f"Testing {op.__name__} with args {args_str}")
        is_correct, error_msg, abs_error, rel_error = eval_correctness_test(op, impl, test)

        verbose_data[args_str] = {
            "correctness_score": 1 if is_correct else 0,
            "correctness_errors": error_msg or "",
            "absolute_error": str(abs_error) if abs_error is not None else "",
            "relative_error": str(rel_error) if rel_error is not None else "",
        }

        if is_correct:
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


def eval_performance(op, impl, tests, verbose_data: defaultdict):
    """Evaluate performance of impl against tests."""
    bench_fn = (
        triton.testing.do_bench if TRITON_AVAILABLE and torch.cuda.is_available() else cpu_bench
    )
    base_times = []
    test_times = []

    for test in tests:
        args_str = serialize_args(test.args, test.kwargs)
        logging.debug(f"Benchmarking {op.__name__} with args {args_str}")
        base_time = bench_fn(lambda: op(*test.args, **test.kwargs))
        base_times.append(base_time)

        try:
            ref = op(*test.args, **test.kwargs)
            res = impl(*test.args, **test.kwargs)
            if not allclose(ref, res):
                raise ValueError(f"Reference and result tensors are not close: {ref} vs {res}")
            test_time = bench_fn(lambda: impl(*test.args, **test.kwargs))
        except Exception:
            test_time = -1

        test_times.append(test_time)
        verbose_data[args_str]["benchmark_time"] = str(test_time)
        speedup = base_time / test_time if test_time > 0 else float("inf")
        verbose_data[args_str]["speedup"] = str(speedup)

    speedups = torch.tensor(base_times) / torch.tensor(test_times)
    return speedups.log().mean().exp()


def eval_one_op(op, impl, correctness_tests, performance_tests):
    """Evaluate impl of op against correctness_tests and performance_tests.

    Returns:
        Tuple of (correctness_score, performance_score, verbose_data)
    """
    verbose_data = defaultdict(dict)

    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        for test in correctness_tests + performance_tests:
            args_str = serialize_args(test.args, test.kwargs)
            verbose_data[args_str] = {
                "correctness_score": 0,
                "benchmark_time": "",
                "speedup": "",
                "correctness_errors": "Skipped: uses CUDA stream",
                "absolute_error": "",
                "relative_error": "",
            }
        return 0, 0, verbose_data

    correctness_score = eval_correctness(op, impl, correctness_tests, verbose_data)
    performance_score = eval_performance(op, impl, performance_tests, verbose_data)
    verbose_data = dict(verbose_data)
    return correctness_score, performance_score, verbose_data


def save_verbose_results(
    results: List[Dict[str, Any]], output_path: str = "backendbench_verbose_results.json"
):
    """Save verbose results to a JSON file."""
    with open(Path(output_path), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Verbose results saved to {output_path}")
