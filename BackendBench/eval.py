# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from collections import defaultdict
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

from BackendBench.utils import serialize_args, uses_cuda_stream, compute_errors

logger = logging.getLogger(__name__)

EXC_MSG = """
Exception raised for {op}:
    args: {args}
    exc: {exc}
"""


def format_exception(e, op, args, kwargs):
    op_name = getattr(op, "__name__", str(op))
    return EXC_MSG.format(op=op_name, args=serialize_args(args, kwargs), exc=e)


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


def eval_correctness(op, impl, tests, test_data: defaultdict = defaultdict(dict)):
    """Evaluate correctness of impl against tests."""
    correct, total = 0, 0
    for test in tests:
        args_str = serialize_args(test.args, test.kwargs)
        logging.debug(f"Testing {op.__name__} with args {args_str}")
        is_correct, error_msg, abs_error, rel_error = eval_correctness_test(op, impl, test)

        test_data[args_str] = {
            "correctness_score": 1 if is_correct else 0,
            "correctness_errors": error_msg or "",
            "absolute_error": str(abs_error) if abs_error is not None else "",
            "relative_error": str(rel_error) if rel_error is not None else "",
        }

        if is_correct:
            correct += 1
        total += 1

    # Handle the case where no tests are available
    if total == 0:
        logger.warning(f"No correctness tests available for {str(op)}")
        return 0.0

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


def eval_performance(op, impl, tests, test_data: defaultdict = defaultdict(dict)):
    """Evaluate performance of impl against tests."""
    bench_fn = (
        triton.testing.do_bench if TRITON_AVAILABLE and torch.cuda.is_available() else cpu_bench
    )
    base_times = []
    test_times = []
    args_strs = []

    for test in tests:
        args_str = serialize_args(test.args, test.kwargs)
        args_strs.append(args_str)
        logging.debug(f"Benchmarking {op.__name__} with args {args_str}")
        base_time = bench_fn(lambda: op(*test.args, **test.kwargs))
        base_times.append(base_time)
        test_time = base_time
        try:
            ref = op(*test.args, **test.kwargs)
            res = impl(*test.args, **test.kwargs)
            if not allclose(
                ref,
                res,
            ):
                raise ValueError(f"Reference and result tensors are not close: {ref} vs {res}")
            test_time = bench_fn(lambda: impl(*test.args, **test.kwargs))
        except Exception:
            pass
        finally:
            test_times.append(test_time)
            test_data[args_str]["benchmark_time"] = str(test_time)

    speedups = torch.tensor(base_times) / torch.tensor(test_times)

    # Update test_data with speedups from the tensor
    for i, args_str in enumerate(args_strs):
        test_data[args_str]["speedup"] = str(speedups[i].item())

    return speedups.log().mean().exp()


def eval_one_op(op, impl, correctness_tests, performance_tests):
    """Evaluate impl of op against correctness_tests and performance_tests.

    Returns:
        Tuple of (correctness_score, performance_score, test_data)
    """
    test_data = defaultdict(dict)

    if uses_cuda_stream(impl):
        logger.warning(f"Skipping {op.__name__} because it uses CUDA stream")
        for test in correctness_tests + performance_tests:
            args_str = serialize_args(test.args, test.kwargs)
            test_data[args_str] = {
                "correctness_score": 0,
                "benchmark_time": "",
                "speedup": "",
                "correctness_errors": "Skipped: uses CUDA stream",
                "absolute_error": "",
                "relative_error": "",
            }
        return 0, 1.0, test_data

    correctness_score = eval_correctness(op, impl, correctness_tests, test_data)
    performance_score = eval_performance(op, impl, performance_tests, test_data)
    test_data = dict(test_data)
    return correctness_score, performance_score, test_data


def generate_and_test_kernel_with_llm_backend(
    backend,
    op,
    test_cases,
    max_attempts=5,
    debug_mode=False,
    debug_op=None
) -> tuple[str, int, bool]:
    """Updated to use conversation manager for LLM backends."""

    # Extract operation name
    op_str = str(op)
    if "aten." in op_str:
        op_name = op_str.split("aten.")[-1].split(".")[0]
    else:
        op_name = op_str.split(".")[-1]

    # Enable debug mode for specific operator
    enable_debug = debug_mode or (debug_op and op_name == debug_op)

    # Check if backend supports conversation history
    if enable_debug and hasattr(backend, 'debug_mode'):
        # Use conversation-aware generation
        logger.info(f"🎯 Using conversation-aware generation for {op_name} (debug mode enabled)")

        # Initialize backend debug mode if not already
        if not backend.debug_mode:
            backend.debug_mode = True
            if hasattr(backend, 'conversations_dir') and not hasattr(backend, '_conversations_dir_created'):
                import os
                backend.conversations_dir = os.path.join(backend.kernels_dir, "conversations")
                os.makedirs(backend.conversations_dir, exist_ok=True)
                backend._conversations_dir_created = True

        # Get the appropriate LLM client
        if backend.name == "llm":
            from BackendBench.llm_client import ClaudeKernelGenerator
            llm_client = ClaudeKernelGenerator()
        elif backend.name == "llm-relay":
            from BackendBench.llm_client import LLMKernelGenerator
            llm_client = LLMKernelGenerator(model=getattr(backend, 'model', 'gcp-claude-4-sonnet'))
        else:
            # Fallback for unsupported backends
            logger.warning(f"Backend {backend.name} does not support conversation history, using legacy approach")
            return _generate_with_legacy_approach(backend, op, op_name, test_cases, max_attempts)

        # Convert test_cases to list to avoid generator exhaustion on subsequent attempts
        test_cases_list = list(test_cases)

        # Create feedback callback
        def feedback_callback(kernel_code: str, attempt: int) -> tuple[bool, Dict]:
            return backend.test_kernel_correctness(op, kernel_code, test_cases_list, attempt)

        # Use conversation-aware generation
        op_signature = f"def {op_name}(*args, **kwargs) -> torch.Tensor"
        op_description = f"PyTorch operation: {op_name}"

        kernel_code, attempts_used, success, conversation_manager = llm_client.generate_kernel_with_conversation_history(
            op_name=op_name,
            op_signature=op_signature,
            op_description=op_description,
            framework="triton",
            max_attempts=max_attempts,
            feedback_callback=feedback_callback,
            debug_mode=enable_debug
        )

        # Store conversation manager in backend for debugging
        backend.conversation_managers[op_name] = conversation_manager

        # Save conversation logs if debug mode enabled
        if enable_debug and hasattr(backend, 'conversations_dir'):
            conversation_manager.save_conversation_log(backend.conversations_dir)

        return kernel_code, attempts_used, success

    else:
        # Use existing approach for backward compatibility
        return _generate_with_legacy_approach(backend, op, op_name, test_cases, max_attempts)


def _generate_with_legacy_approach(backend, op, op_name: str, test_cases, max_attempts: int) -> tuple[str, int, bool]:
    """Legacy approach for backends that don't support conversation history."""
    logger.info(f"Using legacy generation for {op_name}")

    # Get the appropriate LLM client based on backend type
    if backend.name == "llm":
        from BackendBench.llm_client import ClaudeKernelGenerator
        llm_client = ClaudeKernelGenerator()
    elif backend.name == "llm-relay":
        from BackendBench.llm_client import LLMKernelGenerator
        llm_client = LLMKernelGenerator(model=getattr(backend, 'model', 'gcp-claude-4-sonnet'))
    else:
        raise ValueError(f"Unsupported backend for LLM generation: {backend.name}")

    # Create feedback callback
    def feedback_callback(kernel_code: str, attempt: int) -> tuple[bool, Dict]:
        return backend.test_kernel_correctness(op, kernel_code, test_cases, attempt)

    # Use existing retry approach
    op_signature = f"def {op_name}(*args, **kwargs) -> torch.Tensor"
    op_description = f"PyTorch operation: {op_name}"

    kernel_code, attempts_used, success = llm_client.generate_kernel_with_retry(
        op_name=op_name,
        op_signature=op_signature,
        op_description=op_description,
        framework="triton",
        max_attempts=max_attempts,
        feedback_callback=feedback_callback,
    )

    return kernel_code, attempts_used, success


def save_verbose_results(
    results: List[Dict[str, Any]],
    output_path: str = "backendbench_verbose_results.json",
):
    """Save verbose results to a JSON file."""
    with open(Path(output_path), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Verbose results saved to {output_path}")
