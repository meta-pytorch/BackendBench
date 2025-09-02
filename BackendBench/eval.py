# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import csv
from dataclasses import dataclass, asdict
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union
import math
from BackendBench.data_loaders import (
    TORCHBENCH_SUITE_HF_COMMIT,
    HUGGINGFACE_REPO,
    TORCHBENCH_SUITE_FILE,
    _get_hf_url,
)

import torch
from BackendBench.utils import compute_errors, serialize_args, uses_cuda_stream


@dataclass
class CorrectnessTestResult:
    op_name: str
    args: str
    is_correct: bool = False
    error_msg: str = ""
    max_abs_error: float = -math.inf
    max_rel_error: float = -math.inf
    test_type: str = "correctness"


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

        abs_error, rel_error = compute_errors(ref, res)

        return is_correct, None, abs_error, rel_error
    except Exception as e:
        error_msg = format_exception(e, op, args, kwargs)
        logger.warning(error_msg)
        return False, str(e), None, None


def eval_correctness(op, impl, tests) -> Tuple[float, List[CorrectnessTestResult]]:
    """Evaluate correctness of impl against tests."""
    correct, total = 0, 0
    test_results: List[CorrectnessTestResult] = []
    for test in tests:
        args_str = serialize_args(test.args, test.kwargs)
        logging.debug(f"Testing {op.__name__} with args {args_str}")
        is_correct, error_msg, abs_error, rel_error = eval_correctness_test(op, impl, test)

        test_results.append(
            CorrectnessTestResult(
                op_name=op.__name__,
                args=args_str,
                is_correct=is_correct,
                error_msg=error_msg,
                max_abs_error=abs_error,
                max_rel_error=rel_error,
            )
        )

        if is_correct:
            correct += 1
        total += 1

    # Handle the case where no tests are available
    if total == 0:
        logger.warning(f"No correctness tests available for {str(op)}")
        return 0.0, []

    return correct / total, test_results


def cpu_bench(fn, num_runs=100):
    """Simple CPU benchmarking using time.perf_counter."""
    import time

    for _ in range(10):
        fn()

    start = time.perf_counter()
    for _ in range(num_runs):
        fn()
    return (time.perf_counter() - start) / num_runs


def eval_performance(op, impl, tests) -> Tuple[float, List[PerformanceTestResult]]:
    """Evaluate performance of impl against tests."""
    bench_fn = (
        triton.testing.do_bench if TRITON_AVAILABLE and torch.cuda.is_available() else cpu_bench
    )
    base_times = []
    test_times = []
    args_strs = []
    performance_results: List[PerformanceTestResult] = []

    for test in tests:
        args_str = serialize_args(test.args, test.kwargs)
        args_strs.append(args_str)
        logging.debug(f"Benchmarking {op.__name__} with args {args_str}")
        base_time = bench_fn(lambda: op(*test.args, **test.kwargs))
        base_times.append(base_time)
        # Note: If the test fails we consider the speedup to be 1.0
        # TODO: We should make this more explicit, by having an if resolving it in the except and removing the finally block
        test_time = base_time
        try:
            ref = op(*test.args, **test.kwargs)
            res = impl(*test.args, **test.kwargs)
            if not allclose(
                ref,
                res,
            ):
                abs_error, rel_error = compute_errors(ref, res)
                raise ValueError(
                    f"Reference and result tensors are not close: max absolute error {abs_error}, max relative error {rel_error}"
                )
            test_time = bench_fn(lambda: impl(*test.args, **test.kwargs))
            performance_results.append(
                PerformanceTestResult(
                    op_name=op.__name__,
                    args=args_str,
                    speedup=test_time / base_time,
                    successfully_ran=True,
                    benchmark_time_ms=test_time,
                    reference_time_ms=base_time,
                )
            )
        except Exception as e:
            error_msg = format_exception(e, op, test.args, test.kwargs)
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
    op, impl, correctness_tests, performance_tests
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
                    is_correct=False,
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

    correctness_score, correctness_results = eval_correctness(op, impl, correctness_tests)
    performance_score, performance_results = eval_performance(op, impl, performance_tests)
    return correctness_score, performance_score, correctness_results, performance_results


def save_results(
    correctness_results: List[CorrectnessTestResult],
    performance_results: List[PerformanceTestResult],
    output_path: Union[str, Path] = "backendbench_output",
    command: str = None,
    mean_correctness: float = None,
    geomean_perf: float = None,
    perf_at_p_score: float = None,
    p: float = 1.0,
    uses_torchbench: bool = False,
):
    """Save results without creating per-operator directories.

    Args:
        correctness_results: List of correctness test results
        performance_results: List of performance test results
        output_path: Base directory for saving results
        command: Command used to run the benchmark
        mean_correctness: Mean correctness score
        geomean_perf: Geometric mean of performance scores
        perf_at_p_score: Performance at threshold p score
        p: The threshold value used for perf@p calculation
        uses_torchbench: Whether the benchmark uses the TorchBench test suite

    Structure created:
        output_path/
        ├── README.md                  # Top level summary of results
        ├── full_results.json          # Complete results log
        ├── operator_summary.csv       # Operator-level summary
        └── failed_ops.json            # Log of failed operations
    """
    base_dir = Path(output_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Prep work: save all results as a list of dicts
    all_results = [asdict(result) for result in correctness_results] + [
        asdict(result) for result in performance_results
    ]

    # sort by op_name, then args
    all_results.sort(key=lambda x: (x["op_name"], x["args"]))

    # 1. Save the full log in the base directory
    full_log_path = base_dir / "full_results.json"
    failed_ops_path = base_dir / "failed_ops.json"
    summary_csv_path = base_dir / "operator_summary.csv"

    with open(full_log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Full results saved to {full_log_path}")

    # 2. Organize results by operator (without creating directories)
    op_all_results = defaultdict(list)
    op_summaries = {}
    failed_tests = []

    for result in correctness_results:
        op_all_results[result.op_name].append(result)
    for result in performance_results:
        op_all_results[result.op_name].append(result)

    # Process each operator for summary
    for op_name, op_tests in op_all_results.items():
        correctness_results = [
            result for result in op_tests if isinstance(result, CorrectnessTestResult)
        ]
        performance_results = [
            result for result in op_tests if isinstance(result, PerformanceTestResult)
        ]

        # Calculate operator-level summary
        total_tests = len(op_tests)
        correct_tests = sum(1 for result in correctness_results if result.is_correct)
        # Collect performance metrics
        speedups = []
        abs_errors = []
        rel_errors = []

        for test in correctness_results:
            # Check for failures
            if not test.is_correct:
                failed_tests.append(asdict(test))

            if test.max_abs_error and test.max_rel_error:
                abs_errors.append(float(test.max_abs_error))
                rel_errors.append(float(test.max_rel_error))

        for test in performance_results:
            if not test.successfully_ran:
                failed_tests.append(asdict(test))

            # Collect metrics
            if test.speedup:
                speedups.append(float(test.speedup))

        # Calculate summary statistics
        correctness_rate = correct_tests / total_tests if total_tests > 0 else 0.0
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
        geomean_speedup = torch.tensor(speedups).log().mean().exp().item() if speedups else 0.0
        max_abs_error = max(abs_errors) if abs_errors else 0.0
        max_rel_error = max(rel_errors) if rel_errors else 0.0

        op_summaries[op_name] = {
            "operator": op_name,
            "total_tests": total_tests,
            "passed_tests": correct_tests,
            "failed_tests": total_tests - correct_tests,
            "correctness_rate": correctness_rate,
            "avg_speedup": avg_speedup,
            "geomean_speedup": geomean_speedup,
            "max_absolute_error": max_abs_error,
            "max_relative_error": max_rel_error,
        }

    # 3. Create operator-level summary CSV
    if len(op_summaries) > 0:
        op_summary_list = list(op_summaries.values())
        fieldnames = list(op_summary_list[0].keys())

        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in op_summaries.values():
                writer.writerow(summary)

        logger.info(f"Operator summary CSV saved to {summary_csv_path}")

    # 4. Save failed operations log
    # sort failed_tests
    failed_tests.sort(key=lambda x: (x["op_name"], x["args"]))

    with open(failed_ops_path, "w") as f:
        json.dump(failed_tests, f, indent=2)
    logger.info(f"Failed operations log saved to {failed_ops_path}")

    # Save README if metrics are provided
    if all(x is not None for x in [command, mean_correctness, geomean_perf, perf_at_p_score]):
        save_readme(
            output_path=base_dir,
            command=command,
            mean_correctness=mean_correctness,
            geomean_perf=geomean_perf,
            perf_at_p_score=perf_at_p_score,
            p=p,
            uses_torchbench=uses_torchbench,
        )

    # Log summary
    logger.info(f"Results saved to directory: {base_dir.absolute()}")


def save_readme(
    output_path: Union[str, Path],
    command: str,
    mean_correctness: float,
    geomean_perf: float,
    perf_at_p_score: float,
    p: float = 1.0,
    uses_torchbench: bool = False,
):
    """Save a README file with run summary and results.

    Args:
        output_path: Directory to save the README in
        command: The command used to run the benchmark
        mean_correctness: Mean correctness score
        geomean_perf: Geometric mean of performance scores
        perf_at_p_score: Performance at threshold p score
        p: The threshold value used for perf@p calculation
        uses_torchbench: Whether the benchmark uses the TorchBench test suite
    """
    base_dir = Path(output_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    readme_path = base_dir / "README.md"

    with open(readme_path, "w") as f:
        f.write("# BackendBench Run Summary\n\n")

        f.write("## Command\n")
        f.write("```bash\n")
        f.write(f"{command}\n")
        f.write("```\n\n")

        f.write("## Results\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Correctness Score | {mean_correctness:.2f} |\n")
        f.write(f"| Performance Score (geomean speedup) | {geomean_perf:.2f} |\n")
        f.write(f"| Perf@{p} Score | {perf_at_p_score:.2f} |\n")
        f.write("\n")

        f.write("### Metric Descriptions\n\n")
        f.write("- **Correctness Score**: Mean pass rate over all operators\n")
        f.write("- **Performance Score**: Geometric mean speedup over all operators\n")
        f.write(f"- **Perf@{p} Score**: Rate of correct samples with a speedup greater than {p}\n")
        f.write("\n")

        f.write("## Output Files\n\n")
        f.write("The following files are saved in this directory:\n\n")
        f.write("- `full_results.json`: Complete test results for all operators\n")
        f.write("- `operator_summary.csv`: Operator-level summary statistics\n")
        f.write("- `failed_ops.json`: Log of failed operations (if any)\n")
        f.write("- `README.md`: This file\n")

        if uses_torchbench:
            f.write("\n\n")
            f.write("## TorchBench Test Suite Source\n\n")
            f.write("This run used the TorchBench test suite from the following commit:\n\n")
            f.write(f"- TorchBench commit: {TORCHBENCH_SUITE_HF_COMMIT}\n")
            f.write(f"- TorchBench repo: {HUGGINGFACE_REPO}\n")
            f.write(f"- TorchBench suite file: {TORCHBENCH_SUITE_FILE}\n")
            f.write(f"You can download the test suite locally from: {_get_hf_url()}")

    logger.info(f"README saved to {readme_path}")


def perf_at_p(correctness, performance, p=1.0):
    assert len(correctness) == len(performance), (
        "correctness and performance must have the same length"
    )
    return (
        torch.where(torch.tensor(correctness).bool(), torch.tensor(performance) > p, 0)
        .float()
        .mean()
    )
