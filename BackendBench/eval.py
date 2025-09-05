# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import logging
import math
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Union

import torch
from BackendBench.utils import compute_errors, serialize_args, uses_cuda_stream


@dataclass
class CorrectnessTestResult:
    op_name: str
    args: str
    is_correct: bool = False
    error_msg: str = ""
    error_type: str = ""
    traceback: str = ""
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
    traceback: {traceback}
"""


def format_exception(e, op, args, kwargs, traceback=None):
    op_name = getattr(op, "__name__", str(op))
    return EXC_MSG.format(op=op_name, args=serialize_args(args, kwargs), exc=e, traceback=traceback)


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


def eval_correctness_test(op, impl, test, op_name=None) -> CorrectnessTestResult:
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
        result = CorrectnessTestResult(
            op_name=op_name or op.__name__,
            args=serialize_args(args, kwargs),
            is_correct=is_correct,
            max_abs_error=abs_error,
            max_rel_error=rel_error,
        )
        return result
    except Exception as e:
        error_msg = format_exception(e, op, args, kwargs, traceback.format_exc())
        result = CorrectnessTestResult(
            op_name=op_name or op.__name__,
            args=serialize_args(args, kwargs),
            is_correct=False,
            error_msg=error_msg,
            error_type=str(type(e)),
            traceback=traceback.format_exc(),
        )
        logger.warning(error_msg)
        return result


def eval_correctness(op, impl, tests, op_name=None) -> Tuple[float, List[CorrectnessTestResult]]:
    """Evaluate correctness of impl against tests."""
    correct, total = 0, 0
    test_results: List[CorrectnessTestResult] = []
    for test in tests:
        args_str = serialize_args(test.args, test.kwargs)
        logging.debug(f"Testing {op.__name__} with args {args_str}")
        result = eval_correctness_test(op, impl, test, op_name)
        test_results.append(result)
        if result.is_correct:
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
    op, impl, correctness_tests, performance_tests, op_name=None
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

    correctness_score, correctness_results = eval_correctness(op, impl, correctness_tests, op_name)
    performance_score, performance_results = eval_performance(op, impl, performance_tests)
    return (
        correctness_score,
        performance_score,
        correctness_results,
        performance_results,
    )


def save_results(
    correctness_results: List[CorrectnessTestResult],
    performance_results: List[PerformanceTestResult],
    output_path: Union[str, Path] = "backendbench_output",
    command: str = None,
    mean_correctness: float = None,
    geomean_perf: float = None,
    perf_at_p_score: float = None,
    p: float = 1.0,
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

    Structure created:
        output_path/
        ├── OVERALL_SUMMARY.md         # Top level summary of results
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

    # 2. Organize results by operator for csv
    op_all_results = defaultdict(list)
    op_summaries = {}

    for result in correctness_results:
        op_all_results[result.op_name].append(result)
    for result in performance_results:
        op_all_results[result.op_name].append(result)

    # Process each operator for summary
    for op_name, op_tests in op_all_results.items():
        op_correctness_results = [
            result for result in op_tests if isinstance(result, CorrectnessTestResult)
        ]
        op_performance_results = [
            result for result in op_tests if isinstance(result, PerformanceTestResult)
        ]

        # Calculate operator-level summary
        total_tests = len(op_tests)
        correct_tests = sum(1 for result in op_correctness_results if result.is_correct)
        # Collect performance metrics
        speedups = []
        abs_errors = []
        rel_errors = []

        # collect metrics
        for test in op_correctness_results:
            if test.max_abs_error and test.max_rel_error:
                abs_errors.append(float(test.max_abs_error))
                rel_errors.append(float(test.max_rel_error))

        for test in op_performance_results:
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
    failed_tests = [asdict(result) for result in correctness_results if not result.is_correct] + [
        asdict(result) for result in performance_results if not result.successfully_ran
    ]
    # sort failed_tests
    failed_tests.sort(key=lambda x: (x["op_name"], x["args"]))

    with open(failed_ops_path, "w") as f:
        json.dump(failed_tests, f, indent=2)
    logger.info(f"Failed operations log saved to {failed_ops_path}")

    # Save overall_summary if metrics are provided
    if all(x is not None for x in [command, mean_correctness, geomean_perf, perf_at_p_score]):
        save_overall_summary(
            output_path=base_dir,
            command=command,
            mean_correctness=mean_correctness,
            geomean_perf=geomean_perf,
            perf_at_p_score=perf_at_p_score,
            p=p,
            performance_results=performance_results,
            correctness_results=correctness_results,
        )

    # Log summary
    logger.info(f"Results saved to directory: {base_dir.absolute()}")
    print(f"Results saved to directory: {base_dir.absolute()}")
    print(f"Overall summary saved to: {base_dir.absolute()}/OVERALL_SUMMARY.md")


def _get_summary_op_results(
    performance_results: List[PerformanceTestResult],
    correctness_results: List[CorrectnessTestResult],
) -> List[Tuple[str, float, float]]:
    """Get the ops and with correectness ratios and average speedups from the results and sort by descending order of speedups. We return these as strings"""

    correctness_results_dict = defaultdict(list)
    speedups_dict = defaultdict(list)
    op_names = set()
    for result in performance_results:
        # as we assume a broken test defaults back to eager, pretend that the speedup is 1.0 for those in the final calculation
        speedup = 1.0 if not result.successfully_ran else result.speedup
        speedups_dict[result.op_name].append(speedup)
        op_names.add(result.op_name)
    for result in correctness_results:
        correctness_results_dict[result.op_name].append(1.0 if result.is_correct else 0.0)
        op_names.add(result.op_name)

    # string formatting
    op_results = []
    for op in op_names:
        if len(correctness_results_dict[op]) > 0:
            correctness = sum(correctness_results_dict[op]) / len(correctness_results_dict[op])
            correctness = f"{correctness:.4f}%"
        else:
            correctness = "N/A"
        if len(speedups_dict[op]) > 0:
            speedup = sum(speedups_dict[op]) / len(speedups_dict[op])
            speedup = f"{speedup:.4f}x"
        else:
            speedup = "N/A"
        op_results.append((op, correctness, speedup))
    # sort by descending order of speedups and ascending order of correctness
    op_results.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return op_results


def save_overall_summary(
    output_path: Union[str, Path],
    command: str,
    mean_correctness: float,
    geomean_perf: float,
    perf_at_p_score: float,
    p: float = 1.0,
    performance_results: List[PerformanceTestResult] = None,
    correctness_results: List[CorrectnessTestResult] = None,
):
    """Save an overall_summary file with run summary and results.

    Args:
        output_path: Directory to save the overall_summary in
        command: The command used to run the benchmark
        mean_correctness: Mean correctness score
        geomean_perf: Geometric mean of performance scores
        perf_at_p_score: Performance at threshold p score
        p: The threshold value used for perf@p calculation
    """
    base_dir = Path(output_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    overall_summary_path = base_dir / "OVERALL_SUMMARY.md"
    op_results = _get_summary_op_results(performance_results, correctness_results)

    with open(overall_summary_path, "w") as f:
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
        f.write("- `OVERALL_SUMMARY.md`: This file\n")

        f.write("### Operator Speedups vs Eager in Descending Order\n\n")
        f.write("| Operator | Correctness Ratio | Speedup vs Eager |\n")
        f.write("|----------|-----------|----------------|\n")
        for op, correctness, speedup in op_results:
            f.write(f"| {op} | {correctness} | {speedup}|\n")
        f.write("\n")

    logger.info(f"Overall summary saved to {overall_summary_path}")


def perf_at_p(correctness, performance, p=1.0):
    assert len(correctness) == len(performance), (
        "correctness and performance must have the same length"
    )
    return (
        torch.where(torch.tensor(correctness).bool(), torch.tensor(performance) > p, 0)
        .float()
        .mean()
    )
