# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a helper script to analyze the test suite and provide statistics about the number of tests per operation.
"""

import statistics

import torch
from BackendBench.facto_suite import FactoTestSuite
from BackendBench.opinfo_suite import OpInfoTestSuite
from BackendBench.torchbench_suite import TorchBenchTestSuite
from BackendBench.scripts.pytorch_operators import extract_operator_name


def analyze_test_suite(suite):
    test_counts = {}
    total_correctness_tests = 0
    total_performance_tests = 0

    print(f"Analyzing suite: {suite.name}")

    for op_test in suite:
        op_str = str(op_test.op)
        op_name = extract_operator_name(op_str)

        num_correctness_tests = 0
        for _ in op_test.correctness_tests:
            num_correctness_tests += 1

        # In case later we have different tests for performance
        num_performance_tests = 0
        for _ in op_test.performance_tests:
            num_performance_tests += 1

        if op_name not in test_counts:
            test_counts[op_name] = {
                "full_op_str": [],
                "num_correctness_tests": 0,
                "num_performance_tests": 0,
            }
        test_counts[op_name]["full_op_str"].append(op_str)
        test_counts[op_name]["num_correctness_tests"] += num_correctness_tests
        test_counts[op_name]["num_performance_tests"] += num_performance_tests

        total_correctness_tests += num_correctness_tests
        total_performance_tests += num_performance_tests
        # print(f"  {op_name}: {num_correctness_tests} correctness tests and {num_performance_tests} performance tests")

    # Calculate statistics
    test_numbers_dict = {
        "correctness": [info["num_correctness_tests"] for info in test_counts.values()],
        "performance": [info["num_performance_tests"] for info in test_counts.values()],
    }

    results = {"operations": test_counts}
    for correct_or_perf, test_numbers in test_numbers_dict.items():
        if correct_or_perf == "correctness":
            total_tests = total_correctness_tests
        elif correct_or_perf == "performance":
            total_tests = total_performance_tests
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
        results[f"{correct_or_perf}_stats"] = stats

    return results


def print_summary(analysis_results, suite_name, correct_or_perf="correctness"):
    """Print a formatted summary of the analysis results."""
    if correct_or_perf not in ["correctness", "performance"]:
        raise ValueError(f"Invalid value for 'correct_or_perf': {correct_or_perf}")
    stats = analysis_results[f"{correct_or_perf}_stats"]
    num_tests_str = f"num_{correct_or_perf}_tests"

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

        for count in test_counts:
            for i in range(len(bins) - 1):
                if bins[i] <= count < bins[i + 1]:
                    distribution[i] += 1
                    break

        print("\nTest Count Distribution:")
        for i, (label, count) in enumerate(zip(bin_labels, distribution)):
            percentage = (count / len(operations)) * 100
            print(f"  {label:>6} tests: {count:3d} operations ({percentage:5.1f}%)")


def main():
    suite_dict = {
        "opinfo": OpInfoTestSuite(
            "opinfo_cuda_bfloat16",
            "cuda",
            torch.bfloat16,
        ),
        "torchbench": TorchBenchTestSuite(
            "torchbench",
        ),
        "facto": FactoTestSuite(
            "facto_cuda_bfloat16",
            "cuda",
            torch.bfloat16,
        ),
    }

    # Analyze each suite
    all_results = {}
    for suite_name, suite in suite_dict.items():
        print(f"\n{'=' * 60}")
        print(f"ANALYZING {suite_name.upper()} SUITE")
        print(f"{'=' * 60}")

        results = analyze_test_suite(suite)
        all_results[suite_name] = results

        print_summary(results, suite_name, "correctness")
        print_summary(results, suite_name, "performance")

    # If analyzing all suites, provide a comparison summary
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("SUITE COMPARISON SUMMARY")
        print(f"{'=' * 60}")

        print(
            f"{'Suite':<15} {'Ops':<8} {'Tests':<8} {'Avg':<8} {'Min':<6} {'Max':<6} {'StdDev':<8}"
        )
        print("-" * 65)

        for correct_or_perf in ["correctness", "performance"]:
            print(f"\nFor {correct_or_perf} tests")
            for suite_name, results in all_results.items():
                stats = results[f"{correct_or_perf}_stats"]
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
    main()
