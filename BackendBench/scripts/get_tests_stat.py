# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a helper script to analyze the test suite and provide statistics about the number of tests per operation.
"""

import click
import statistics
import traceback

import torch
from BackendBench.facto_suite import FactoTestSuite
from BackendBench.suite import SmokeTestSuite
from BackendBench.opinfo_suite import OpInfoTestSuite
from BackendBench.torchbench_suite import DEFAULT_HUGGINGFACE_URL, TorchBenchTestSuite
from BackendBench.scripts.pytorch_operators import extract_operator_name


def analyze_test_suite(suite):
    test_counts = {}
    total_tests = 0
    total_perf_tests = 0

    print(f"Analyzing suite: {suite.name}")

    for op_test in suite:
        op_str = str(op_test.op)
        op_name = extract_operator_name(op_str)

        num_tests = 0
        for _ in op_test.correctness_tests:
            num_tests += 1

        # In case later we have different tests for performance
        num_perf_tests = 0
        for _ in op_test.performance_tests:
            num_perf_tests += 1

        test_counts[op_name] = {
            "full_op_name": op_str,
            "num_tests": num_tests,
            "num_perf_tests": num_perf_tests,
        }

        total_tests += num_tests
        total_perf_tests += num_perf_tests
        # print(f"  {op_name}: {num_tests} correctness tests and {num_perf_tests} performance tests")

    # Calculate statistics
    test_numbers = [info["num_tests"] for info in test_counts.values()]
    perf_test_numbers = [info["num_perf_tests"] for info in test_counts.values()]

    if test_numbers:
        stats = {
            "total_operations": len(test_counts),
            "total_tests": total_tests,
            "min_tests": min(test_numbers),
            "max_tests": max(test_numbers),
            "mean_tests": statistics.mean(test_numbers),
            "median_tests": statistics.median(test_numbers),
            "stdev_tests": statistics.stdev(test_numbers) if len(test_numbers) > 1 else 0.0,
        }
    else:
        stats = {
            "total_operations": 0,
            "total_tests": 0,
            "min_tests": 0,
            "max_tests": 0,
            "mean_tests": 0.0,
            "median_tests": 0.0,
            "stdev_tests": 0.0,
        }

    if perf_test_numbers:
        perf_stats = {
            "total_operations": len(test_counts),
            "total_tests": total_perf_tests,
            "min_tests": min(perf_test_numbers),
            "max_tests": max(perf_test_numbers),
            "mean_tests": statistics.mean(perf_test_numbers),
            "median_tests": statistics.median(perf_test_numbers),
            "stdev_tests": statistics.stdev(perf_test_numbers)
            if len(perf_test_numbers) > 1
            else 0.0,
        }
    else:
        perf_stats = {
            "total_operations": 0,
            "total_tests": 0,
            "min_tests": 0,
            "max_tests": 0,
            "mean_tests": 0.0,
            "median_tests": 0.0,
            "stdev_tests": 0.0,
        }

    return {"stats": stats, "perf_stats": perf_stats, "operations": test_counts}


def print_summary(analysis_results, suite_name, correct_or_perf="correctness"):
    """Print a formatted summary of the analysis results."""
    if correct_or_perf == "correctness":
        stats = analysis_results["stats"]
        num_tests_str = "num_tests"
    elif correct_or_perf == "performance":
        stats = analysis_results["perf_stats"]
        num_tests_str = "num_perf_tests"
    else:
        raise ValueError(f"Invalid value for 'correct_or_perf': {correct_or_perf}")

    operations = analysis_results["operations"]

    print(f"\n{'=' * 60}")
    print(f"TEST STATISTICS SUMMARY FOR {suite_name.upper()} SUITE {correct_or_perf.upper()} TESTS")
    print(f"{'=' * 60}")

    print("\nOverall Statistics:")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Total tests: {stats['total_tests']}")
    print(f"  Average tests per operation: {stats['mean_tests']:.2f}")
    print(f"  Median tests per operation: {stats['median_tests']:.2f}")
    print(f"  Min tests per operation: {stats['min_tests']}")
    print(f"  Max tests per operation: {stats['max_tests']}")
    print(f"  Standard deviation: {stats['stdev_tests']:.2f}")

    if operations:
        test_counts = [info[num_tests_str] for info in operations.values()]
        bins = [1, 5, 10, 25, 50, 100, float("inf")]
        bin_labels = ["1-4", "5-9", "10-24", "25-49", "50-99", "100+"]
        distribution = [0] * (len(bins))

        count_1_tests = 0
        for count in test_counts:
            if count == 1:
                count_1_tests += 1
            for i in range(len(bins) - 1):
                if bins[i] <= count < bins[i + 1]:
                    distribution[i] += 1
                    break

        print("\nTest Count Distribution:")
        print(f"  Operations with 1 test: {count_1_tests}")
        for i, (label, count) in enumerate(zip(bin_labels, distribution)):
            percentage = (count / len(operations)) * 100
            print(f"  {label:>6} tests: {count:3d} operations ({percentage:5.1f}%)")


@click.command()
@click.option(
    "--suite",
    default="opinfo",
    type=click.Choice(["smoke", "opinfo", "torchbench", "facto", "all"]),
    help="Which suite to analyze",
)
@click.option(
    "--ops",
    default=None,
    type=str,
    help="Comma-separated list of ops to run",
)
@click.option(
    "--topn-inputs",
    "--topn",
    default=None,
    type=int,
    help="Select the top N largest inputs for each op (default: all inputs)",
)
@click.option(
    "--torchbench-data-path",
    default=DEFAULT_HUGGINGFACE_URL,
    type=str,
    help="Path to TorchBench operator data",
)
def cli(
    suite,
    ops,
    topn_inputs,
    torchbench_data_path,
):
    if ops:
        ops = ops.split(",")
    suites_to_analyze = []

    suite_dict = {
        "smoke": lambda: SmokeTestSuite,
        "opinfo": lambda: OpInfoTestSuite(
            "opinfo_cuda_bfloat16",
            "cuda",
            torch.bfloat16,
            filter=ops,
        ),
        "torchbench": lambda: TorchBenchTestSuite(
            "torchbench",
            torchbench_data_path,
            filter=ops,
            topn=topn_inputs,
        ),
        "facto": lambda: FactoTestSuite(
            "facto_cuda_bfloat16",
            "cuda",
            torch.bfloat16,
            filter=ops,
        ),
    }

    if suite == "all":
        for suite_name, suite_factory in suite_dict.items():
            suites_to_analyze.append((suite_name, suite_factory()))
    else:
        suites_to_analyze.append((suite, suite_dict[suite]()))

    # Analyze each suite
    all_results = {}
    for suite_name, suite in suites_to_analyze:
        try:
            print(f"\n{'=' * 60}")
            print(f"ANALYZING {suite_name.upper()} SUITE")
            print(f"{'=' * 60}")

            results = analyze_test_suite(suite)
            all_results[suite_name] = results

            print_summary(results, suite_name, "correctness")
            print_summary(results, suite_name, "performance")

        except Exception as e:
            print(f"Error analyzing {suite_name} suite: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            continue

    # If analyzing all suites, provide a comparison summary
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("SUITE COMPARISON SUMMARY")
        print(f"{'=' * 60}")

        print(
            f"{'Suite':<15} {'Ops':<8} {'Tests':<8} {'Avg':<8} {'Min':<6} {'Max':<6} {'StdDev':<8}"
        )
        print("-" * 65)

        print("\nFor correctness tests")
        for suite_name, results in all_results.items():
            stats = results["stats"]
            print(
                f"{suite_name:<15} "
                f"{stats['total_operations']:<8} "
                f"{stats['total_tests']:<8} "
                f"{stats['mean_tests']:<8.1f} "
                f"{stats['min_tests']:<6} "
                f"{stats['max_tests']:<6} "
                f"{stats['stdev_tests']:<8.1f}"
            )

        print("\nFor performance tests")
        for suite_name, results in all_results.items():
            stats = results["perf_stats"]
            print(
                f"{suite_name:<15} "
                f"{stats['total_operations']:<8} "
                f"{stats['total_tests']:<8} "
                f"{stats['mean_tests']:<8.1f} "
                f"{stats['min_tests']:<6} "
                f"{stats['max_tests']:<6} "
                f"{stats['stdev_tests']:<8.1f}"
            )


if __name__ == "__main__":
    cli()
