# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import traceback
from dataclasses import dataclass
from typing import List, Tuple
import time

import torch

from BackendBench.utils import compute_errors, serialize_args, uses_cuda_stream
from collections import defaultdict

# todo: Eventaully we should move whether or not we test the backwards pass to the suite
BACKWARDS_PASS_TESTING_EXCEMPTIONS = [
    # We skip this op for 2 reasons.
    # 1) This op has the args (shape, stride, storage_offset) where storage offset would change if a gradient is included in the tensor. 
    # Our suites (ie. opinfo) assumes we are doing inference so storage is set to a bad value here. We'd have to write a custom suite for this
    # 2) As this is a tensor manipulation op, it doesn't really make sense to test a backwards pass for this yet.
    "as_strided.default", 
]

# delete this
REF_TYPES = defaultdict(lambda: [])

@dataclass
class CorrectnessTestResult:
    op_name: str
    args: str
    has_correct_output: bool = False
    error_msg: str = ""
    error_type: str = ""
    traceback: str = ""
    max_abs_error: float = -math.inf
    max_rel_error: float = -math.inf
    test_type: str = "correctness"
    has_correct_gradients: bool = False
    checked_backwards: bool = False
    non_non_grads: int = 0
    requires_grads_count: int = 0

@dataclass
class PerformanceTestResult:
    op_name: str
    args: str
    speedup: float
    benchmark_time_ms: float
    reference_time_ms: float
    error_msg: str = ""
    successfully_ran: bool = False
    test_type: str = "performance"


try:
    if torch.cuda.is_available():
        import triton.testing

        TRITON_AVAILABLE = True
    else:
        TRITON_AVAILABLE = False
except ImportError:
    TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)

EXC_MSG = """
Exception raised for {op}:
    args: {args}
    exc: {exc}
    traceback: {traceback}
"""


def format_exception(e, op, args, kwargs, traceback=None):
    op_name = getattr(op, "__name__", str(op))
    return EXC_MSG.format(op=op_name, args=serialize_args(args, kwargs), exc=e, traceback=traceback)

def compare_gradients(res_grad, ref_grad, atol=1e-2, rtol=1e-2):
    if res_grad is None and ref_grad is None:
        return True
    if res_grad is None or ref_grad is None:
        raise ValueError(f"One of the gradients is None while the other is not.")
    return allclose(res_grad, ref_grad, atol=atol, rtol=rtol)

def _apply_to_tensors(obj, tensor_fn, container_fn=None, accumulator=None):
    """
    Generic functor to apply operations to tensors in nested data structures.

    Args:
        obj: The object to traverse (tensor, list, tuple, dict, or other)
        tensor_fn: Function to apply to each tensor. Should have signature (tensor, accumulator) -> Any
        container_fn: Optional function to handle container reconstruction.
                     Signature: (container_type, transformed_items) -> Any
        accumulator: Optional accumulator object passed to tensor_fn

    Returns:
        Transformed object or None for in-place operations
    """
    if isinstance(obj, torch.Tensor):
        return tensor_fn(obj, accumulator)
    elif isinstance(obj, list):
        transformed = [_apply_to_tensors(item, tensor_fn, container_fn, accumulator) for item in obj]
        return container_fn(list, transformed) if container_fn else transformed
    elif isinstance(obj, tuple):
        transformed = [_apply_to_tensors(item, tensor_fn, container_fn, accumulator) for item in obj]
        return container_fn(tuple, transformed) if container_fn else tuple(transformed)
    elif isinstance(obj, dict):
        transformed = {key: _apply_to_tensors(value, tensor_fn, container_fn, accumulator)
                     for key, value in obj.items()}
        return container_fn(dict, transformed) if container_fn else transformed
    else:
        # For immutable types or unknown types
        return obj


def collect_gradients(args, kwargs) -> List[torch.Tensor]:
    """
    Collect all gradients from args and kwargs into a flat list.

    Order is well-defined:
    1. Iterate through args in order
       - If arg is a tensor with grad, append grad
       - If arg is a list/tuple, iterate through elements in order and append tensor grads
    2. Iterate through kwargs in sorted key order
       - If kwarg is a tensor with grad, append grad
       - If kwarg is a list/tuple, iterate through elements in order and append tensor grads

    Args:
        args: The arguments (can contain tensors or lists/tuples of tensors).
        kwargs: The keyword arguments (can contain tensors or lists/tuples of tensors).

    Returns:
        List of gradients (torch.Tensor) in the order specified above.
        Returns empty list if no gradients are found.
    """
    gradients = []

    def collect_grad_fn(tensor, accumulator):
        if tensor.grad is not None:
            accumulator.append(tensor.grad)

    # Collect from args
    for arg in args:
        _apply_to_tensors(arg, collect_grad_fn, accumulator=gradients)

    # Collect from kwargs in sorted key order for deterministic ordering
    for key in sorted(kwargs.keys()):
        _apply_to_tensors(kwargs[key], collect_grad_fn, accumulator=gradients)

    return gradients


def check_input_gradients(res_args, res_kwargs, ref_args, ref_kwargs, atol=1e-2, rtol=1e-2) -> bool:
    """
    Check if the gradients of the result and reference are close.
    Args:
        res_args: The arguments of the result.
        res_kwargs: The keyword arguments of the result.
        ref_args: The arguments of the reference.
        ref_kwargs: The keyword arguments of the reference.
        atol: The absolute tolerance.
        rtol: The relative tolerance.
    Returns:
        True if the gradients are close, False otherwise. If there are no gradients, return True.
    """
    res_grads = collect_gradients(res_args, res_kwargs)
    ref_grads = collect_gradients(ref_args, ref_kwargs)

    if len(res_grads) != len(ref_grads):
        raise ValueError(f"The number of gradients is not the same. {len(res_grads)} vs {len(ref_grads)}")

    for res_grad, ref_grad in zip(res_grads, ref_grads):
        if (a is None) ^ (b is None):
            return False
        elif a is None and b is None:
            continue
        if not allclose(res_grad, ref_grad, atol=atol, rtol=rtol):
            return False

    return True
    
def _allclose(a, b, atol=1e-2, rtol=1e-2):
    # using a stack to avoid recursion overflow issues
    stack = [(a, b)]

    while len(stack) > 0:
        curr_a, curr_b = stack.pop()

        if isinstance(curr_a, torch.Tensor):
            torch.testing.assert_close(curr_a, curr_b, equal_nan=True, atol=atol, rtol=rtol)
        elif isinstance(curr_a, (list, tuple)):
            assert len(curr_a) == len(curr_b)
            # Add pairs to stack in reverse order to maintain left-to-right checking
            stack.extend(reversed(list(zip(curr_a, curr_b))))
        else:
            assert curr_a == curr_b


def allclose(a, b, atol=1e-2, rtol=1e-2):
    try:
        _allclose(a, b)
        return True
    except Exception:
        return False


def make_tensors_require_gradients(args, kwargs):
    def make_require_grad_fn(tensor, _):
        tensor.requires_grad = True
    _apply_to_tensors(args, make_require_grad_fn)
    _apply_to_tensors(kwargs, make_require_grad_fn)


def clear_gradients(args, kwargs):
    def clear_grad_fn(tensor, _):
        if tensor.grad is not None:
            tensor.grad = None
    _apply_to_tensors(args, clear_grad_fn)
    _apply_to_tensors(kwargs, clear_grad_fn)

def count_requires_grad(args, kwargs):
    def check_requires_grad(tensor, _):
        if tensor.requires_grad:
            return 1
        return 0
    accumulator_args = []
    accumulator_kwargs = []
    _apply_to_tensors(args, check_requires_grad, accumulator=accumulator_args)
    _apply_to_tensors(kwargs, check_requires_grad, accumulator=accumulator_kwargs)
    return sum(accumulator_args) + sum(accumulator_kwargs)

def check_if_has_backwards(obj):
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return all(check_if_has_backwards(x) for x in obj) and len(obj) > 0
    else:
        return False

def eval_correctness_test(op, impl, test, check_backwards=False) -> CorrectnessTestResult:
    """Evaluate impl of op against test.

    Returns:
        Tuple of (is_correct, error_message, absolute_error, relative_error)
    """

    # do not check backwards if the op is in our testing excemptions
    # it's important to do this check before adding gradients as we expempt things as adding a gradient would make the forward pass invalid (like as_strided)
    if op.__name__ in BACKWARDS_PASS_TESTING_EXCEMPTIONS:
        check_backwards=False

    args, kwargs = test.args, test.kwargs
    if check_backwards:
        make_tensors_require_gradients(args, kwargs)
    ref = op(*args, **kwargs)
    # REF_TYPES[type(ref)].append(op.__name__)
    # print(f" ref types types are currently {REF_TYPES.keys()}")

    # we now modify check_backwards with another check. Specifically that ref is something that has gradients (aka returns a torch.tensor or a collection of torch.tensors as we cannot perform a backwards pass otherwise)
    backwards_possible = check_if_has_backwards(ref)

    # I'm a bit paranoid as we test more things we might run into something like this
    if not backwards_possible and (isinstance(ref, list) or isinstance(ref, tuple)):
        type_str = f"{[type(x) for x in ref]}"
        raise ValueError(f"{op.__name__} returns a {type(ref)} with elements of types {type_str}.\n At the moment backendbench only supports backwards passes if the collection only contains torch.tensors. Please reach out to the backendbench maintainers to fix this corner case.")
    check_backwards = backwards_possible and check_backwards

    if check_backwards:
            loss = ref.sum()
            loss.backward()
            ref_grads = collect_gradients(args, kwargs)
            clear_gradients(args, kwargs)
    else:
        ref_grads = None

    try:

        res = impl(*args, **kwargs)
        if check_backwards and isinstance(ref, torch.Tensor):
            loss = res.sum()
            loss.backward()
            res_grads = collect_gradients(args, kwargs)
            clear_gradients(args, kwargs)
            has_correct_gradients = compare_gradients(ref_grads, res_grads)
            non_non_grads = len([g for g in res_grads if g is not None])
        else:
            res_grads = None
            has_correct_gradients = False
            non_non_grads = 0
        has_correct_output = allclose(ref, res)
        requires_grads_count = count_requires_grad(args, kwargs)


        abs_error, rel_error = compute_errors(ref, res)
        result = CorrectnessTestResult(
            op_name=op.__name__,
            args=serialize_args(args, kwargs),
            has_correct_output=has_correct_output,
            max_abs_error=abs_error,
            max_rel_error=rel_error,
            has_correct_gradients=has_correct_gradients,
            checked_backwards=check_backwards,
            non_non_grads=non_non_grads,
            requires_grads_count=requires_grads_count,
        )
        return result
    except Exception as e:
        error_msg = format_exception(e, op, args, kwargs, traceback.format_exc())
        result = CorrectnessTestResult(
            op_name=op.__name__,
            args=serialize_args(args, kwargs),
            has_correct_output=False,
            error_msg=error_msg,
            error_type=str(type(e)),
            traceback=traceback.format_exc(),
        )
        logger.warning(error_msg)
        return result


def eval_correctness(op, impl, tests, check_backwards=False) -> Tuple[float, List[CorrectnessTestResult]]:
    """Evaluate correctness of impl against tests."""
    correct, total = 0, 0
    test_results: List[CorrectnessTestResult] = []
    for test in tests:
        args_str = serialize_args(test.args, test.kwargs)
        logging.debug(f"Testing {op.__name__} with args {args_str}")
        result = eval_correctness_test(op, impl, test, check_backwards)
        test_results.append(result)
        if result.has_correct_output:
            correct += 1
        total += 1

    # Handle the case where no tests are available
    if total == 0:
        logger.warning(f"No correctness tests available for {str(op)}")
        return 0.0, []

    return correct / total, test_results


def cpu_bench(fn, num_runs=100):
    """Simple CPU benchmarking using time.perf_counter."""

    for _ in range(10):
        fn()

    start = time.perf_counter()
    for _ in range(num_runs):
        fn()
    return (time.perf_counter() - start) / num_runs


def eval_performance(op, impl, tests) -> Tuple[float, List[PerformanceTestResult]]:
    """Evaluate performance of impl against tests."""
    if TRITON_AVAILABLE and torch.cuda.is_available():
        bench_fn = lambda fn: triton.testing.do_bench(fn, warmup=25, rep=100)
    else:
        bench_fn = cpu_bench
    base_times = []
    test_times = []
    args_strs = []
    performance_results: List[PerformanceTestResult] = []

    for test in tests:
        # Cache the arguments to ensure consistency between reference and implementation
        cached_args = test.args
        cached_kwargs = test.kwargs
        args_str = serialize_args(cached_args, cached_kwargs)
        args_strs.append(args_str)
        logging.debug(f"Benchmarking {op.__name__} with args {args_str}")
        # Warmup: run both operations to compile CUDA kernels and warm up caches
        for _ in range(25):
            _ = op(*cached_args, **cached_kwargs)
            _ = impl(*cached_args, **cached_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        base_time = bench_fn(lambda: op(*cached_args, **cached_kwargs))
        base_times.append(base_time)
        # Note: If the test fails we consider the speedup to be 1.0
        # TODO: We should make this more explicit, by having an if resolving it in the except and removing the finally block
        test_time = base_time
        try:
            ref = op(*cached_args, **cached_kwargs)
            res = impl(*cached_args, **cached_kwargs)
            if not allclose(
                ref,
                res,
            ):
                abs_error, rel_error = compute_errors(ref, res)
                raise ValueError(
                    f"Reference and result tensors are not close: max absolute error {abs_error}, max relative error {rel_error}"
                )
            # Warmup impl again before benchmarking
            for _ in range(25):
                _ = impl(*cached_args, **cached_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            test_time = bench_fn(lambda: impl(*cached_args, **cached_kwargs))
            performance_results.append(
                PerformanceTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    speedup=base_time / test_time,
                    successfully_ran=True,
                    benchmark_time_ms=test_time,
                    reference_time_ms=base_time,
                )
            )
        except Exception as e:
            error_msg = format_exception(e, op, test.args, test.kwargs, traceback.format_exc())
            performance_results.append(
                PerformanceTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    successfully_ran=False,
                    speedup=None,
                    benchmark_time_ms=None,
                    reference_time_ms=base_time,
                    error_msg=error_msg,
                )
            )
        finally:
            test_times.append(test_time)

    speedups = torch.tensor(base_times) / torch.tensor(test_times)

    return speedups.log().mean().exp(), performance_results


def eval_one_op(
    op, impl, correctness_tests, performance_tests, check_backwards=False
) -> Tuple[float, float, List[CorrectnessTestResult], List[PerformanceTestResult]]:
    """Evaluate impl of op against correctness_tests and performance_tests.

    Returns:
        Tuple of (correctness_score, performance_score, correctness_results, performance_results)
    """

    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        performance_results = []
        correctness_results = []
        for test in correctness_tests:
            args_str = serialize_args(test.args, test.kwargs)
            correctness_results.append(
                CorrectnessTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    has_correct_output=False,
                    error_msg="Skipped: uses CUDA stream",
                )
            )
        for test in performance_tests:
            args_str = serialize_args(test.args, test.kwargs)
            performance_results.append(
                PerformanceTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    speedup=0,
                    benchmark_time_ms=0,
                    reference_time_ms=0,
                    error_msg="Skipped: uses CUDA stream",
                )
            )
        return 0, 1.0, correctness_results, performance_results

    correctness_score, correctness_results = eval_correctness(op, impl, correctness_tests, check_backwards)
    performance_score, performance_results = eval_performance(op, impl, performance_tests)
    return (
        correctness_score,
        performance_score,
        correctness_results,
        performance_results,
    )


def perf_at_p(correctness, performance, p=1.0):
    assert len(correctness) == len(performance), (
        "correctness and performance must have the same length"
    )
    return (
        torch.where(torch.tensor(correctness).bool(), torch.tensor(performance) > p, 0)
        .float()
        .mean()
    )
