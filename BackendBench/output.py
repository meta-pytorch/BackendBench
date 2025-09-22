# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Union

import torch

from .eval import CorrectnessTestResult, PerformanceTestResult

logger = logging.getLogger(__name__)


def _prepare_results_data(
    correctness_results: List[CorrectnessTestResult],
    performance_results: List[PerformanceTestResult],
) -> Tuple[List[dict], List[dict], dict]:
    """Prepare and process results data without file I/O.

    Returns:
        Tuple of (all_results, failed_tests, op_summaries)
    """
    # Prep work: save all results as a list of dicts
    all_results = [asdict(result) for result in correctness_results] + [
        asdict(result) for result in performance_results
    ]

    # sort by op_name, then args
    all_results.sort(key=lambda x: (x["op_name"], x["args"]))

    # Organize results by operator for csv
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
        correct_correctness_tests = sum(1 for result in op_correctness_results if result.is_correct)
        passed_performance_tests = sum(
            1 for result in op_performance_results if result.successfully_ran
        )
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
        correctness_rate = (
            correct_correctness_tests / len(op_correctness_results)
            if len(op_correctness_results) > 0
            else 0.0
        )
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
        geomean_speedup = torch.tensor(speedups).log().mean().exp().item() if speedups else 0.0
        max_abs_error = max(abs_errors) if abs_errors else 0.0
        max_rel_error = max(rel_errors) if rel_errors else 0.0

        op_summaries[op_name] = {
            "operator": op_name,
            "total_tests": len(op_all_results),
            "correctness_tests": len(op_correctness_results),
            "performance_tests": len(op_performance_results),
            "passed_correctness_tests": correct_correctness_tests,
            "passed_performance_tests": passed_performance_tests,
            "failed_correctness_tests": len(op_correctness_results) - correct_correctness_tests,
            "failed_performance_tests": len(op_performance_results) - passed_performance_tests,
            "correctness_rate": correctness_rate,
            "avg_speedup": avg_speedup,
            "geomean_speedup": geomean_speedup,
            "max_absolute_error": max_abs_error,
            "max_relative_error": max_rel_error,
        }

    # Prepare failed operations log
    failed_tests = [asdict(result) for result in correctness_results if not result.is_correct] + [
        asdict(result) for result in performance_results if not result.successfully_ran
    ]

    # sort failed_tests
    failed_tests.sort(key=lambda x: (x["op_name"], x["args"]))

    return all_results, failed_tests, op_summaries


def save_results(
    correctness_results: List[CorrectnessTestResult],
    performance_results: List[PerformanceTestResult],
    output_path: str,
    command: str,
    mean_correctness: float,
    geomean_perf: float,
    perf_at_p_score: float,
    p: float = 1.0,
) -> Tuple[List[dict], List[dict], dict]:
    """Prepare and process results data without file I/O.

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
        └── failed_tests.json            # Log of failed operations
    """
    base_dir = Path(output_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Process data using the extracted function
    all_results, failed_tests, op_summaries = _prepare_results_data(
        correctness_results, performance_results
    )

    # 1. Save the full log in the base directory
    full_log_path = os.path.join(base_dir, "full_results.json")
    failed_tests_path = os.path.join(base_dir, "failed_tests.json")
    summary_csv_path = os.path.join(base_dir, "operator_summary.csv")

    with open(full_log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Full results saved to {full_log_path}")

    # 2. Create operator-level summary CSV
    if len(op_summaries) > 0:
        op_summary_list = list(op_summaries.values())
        fieldnames = list(op_summary_list[0].keys())

        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in op_summaries.values():
                writer.writerow(summary)

        logger.info(f"Operator summary CSV saved to {summary_csv_path}")

    # 3. Save failed operations log
    with open(failed_tests_path, "w") as f:
        json.dump(failed_tests, f, indent=2)
    logger.info(f"Failed operations log saved to {failed_tests_path}")

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


def _generate_overall_summary_content(
    command: str,
    mean_correctness: float,
    geomean_perf: float,
    perf_at_p_score: float,
    p: float = 1.0,
    performance_results: List[PerformanceTestResult] = None,
    correctness_results: List[CorrectnessTestResult] = None,
) -> str:
    """Generate the content for the overall summary markdown file.

    Returns:
        The markdown content as a string.
    """
    op_results = _get_summary_op_results(performance_results, correctness_results)

    content = []
    content.append("# BackendBench Run Summary\n")

    content.append("## Command")
    content.append("```bash")
    content.append(f"{command}")
    content.append("```\n")

    content.append("## Results\n")
    content.append("| Metric | Value |")
    content.append("|--------|-------|")
    content.append(f"| Correctness Score | {mean_correctness:.2f} |")
    content.append(f"| Performance Score (geomean speedup) | {geomean_perf:.2f} |")
    content.append(f"| Perf@{p} Score | {perf_at_p_score:.2f} |")
    content.append("")

    content.append("### Metric Descriptions\n")
    content.append("- **Correctness Score**: Mean pass rate over all operators")
    content.append("- **Performance Score**: Geometric mean speedup over all operators")
    content.append(f"- **Perf@{p} Score**: Rate of correct samples with a speedup greater than {p}")
    content.append("")

    content.append("## Output Files\n")
    content.append("The following files are saved in this directory:\n")
    content.append("- `full_results.json`: Complete test results for all operators")
    content.append("- `operator_summary.csv`: Operator-level summary statistics")
    content.append("- `failed_tests.json`: Log of failed tests (if any)")
    content.append("- `OVERALL_SUMMARY.md`: This file")

    content.append("### Operator Speedups vs Eager in Descending Order\n")
    content.append("| Operator | Correctness Ratio | Speedup vs Eager |")
    content.append("|----------|-----------|----------------|")
    for op, correctness, speedup in op_results:
        content.append(f"| {op} | {correctness} | {speedup}|")
    content.append("")

    return "\n".join(content)


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

    overall_summary_path = os.path.join(base_dir, "OVERALL_SUMMARY.md")

    content = _generate_overall_summary_content(
        command,
        mean_correctness,
        geomean_perf,
        perf_at_p_score,
        p,
        performance_results,
        correctness_results,
    )

    with open(overall_summary_path, "w") as f:
        f.write(content)

    logger.info(f"Overall summary saved to {overall_summary_path}")
