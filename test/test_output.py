# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import tempfile
from pathlib import Path

from expecttest import assert_expected_inline

from BackendBench.eval import CorrectnessTestResult, PerformanceTestResult
from BackendBench.output import (
    _get_summary_op_results,
    _prepare_results_data,
    save_overall_summary,
    save_results,
)


class TestOutputFunctions:
    def _create_test_fixtures(self):
        """Create test fixtures for correctness and performance results."""
        correctness_results = [
            CorrectnessTestResult(
                op_name="torch.ops.aten.add.Tensor",
                args="[tensor([1, 2]), tensor([3, 4])]",
                is_correct=True,
                max_abs_error=0.001,
                max_rel_error=0.0001,
            ),
            CorrectnessTestResult(
                op_name="torch.ops.aten.add.Tensor",
                args="[tensor([5, 6]), tensor([7, 8])]",
                is_correct=True,
                max_abs_error=0.002,
                max_rel_error=0.0002,
            ),
            CorrectnessTestResult(
                op_name="torch.ops.aten.mul.Tensor",
                args="[tensor([1, 2]), tensor([3, 4])]",
                is_correct=False,
                error_msg="Tensor mismatch",
                error_type="AssertionError",
            ),
            CorrectnessTestResult(
                op_name="torch.ops.aten.sin.default",
                args="[tensor([0.5])]",
                is_correct=True,
                max_abs_error=0.0,
                max_rel_error=0.0,
            ),
        ]

        performance_results = [
            PerformanceTestResult(
                op_name="torch.ops.aten.add.Tensor",
                args="[tensor([1, 2]), tensor([3, 4])]",
                speedup=1.5,
                benchmark_time_ms=10.0,
                reference_time_ms=15.0,
                successfully_ran=True,
            ),
            PerformanceTestResult(
                op_name="torch.ops.aten.add.Tensor",
                args="[tensor([5, 6]), tensor([7, 8])]",
                speedup=2.0,
                benchmark_time_ms=8.0,
                reference_time_ms=16.0,
                successfully_ran=True,
            ),
            PerformanceTestResult(
                op_name="torch.ops.aten.mul.Tensor",
                args="[tensor([1, 2]), tensor([3, 4])]",
                speedup=0.0,
                benchmark_time_ms=0.0,
                reference_time_ms=20.0,
                successfully_ran=True,
            ),
            PerformanceTestResult(
                op_name="torch.ops.aten.sin.default",
                args="[tensor([0.5])]",
                speedup=None,
                benchmark_time_ms=None,
                reference_time_ms=20.0,
                successfully_ran=False,
                error_msg="Compilation failed",
            ),
        ]

        return correctness_results, performance_results

    def test_prepare_results_data(self):
        """Test the _prepare_results_data function."""
        correctness_results, performance_results = self._create_test_fixtures()

        all_results, failed_tests, op_summaries = _prepare_results_data(
            correctness_results, performance_results
        )

        # Check that all results are properly converted to dicts and sorted
        assert len(all_results) == 8  # 4 correctness + 4 performance

        # Check failed tests
        assert len(failed_tests) == 2  # 1 correctness + 1 performance failure
        failed_tests = [test["op_name"] for test in failed_tests]
        assert "torch.ops.aten.mul.Tensor" in failed_tests
        assert "torch.ops.aten.sin.default" in failed_tests

        # Check operator summaries
        assert len(op_summaries) == 3  # add, mul, sin

        # Test add operator summary
        add_summary = op_summaries["torch.ops.aten.add.Tensor"]
        assert_expected_inline(
            str(add_summary),
            """{'operator': 'torch.ops.aten.add.Tensor', 'total_tests': 3, 'correctness_tests': 2, 'performance_tests': 2, 'passed_correctness_tests': 2, 'passed_performance_tests': 2, 'failed_correctness_tests': 0, 'failed_performance_tests': 0, 'correctness_rate': 1.0, 'avg_speedup': 1.75, 'geomean_speedup': 1.7320507764816284, 'max_absolute_error': 0.002, 'max_relative_error': 0.0002}""",
        )

        # Test mul operator summary (should have failed correctness and performance)
        mul_summary = op_summaries["torch.ops.aten.mul.Tensor"]
        assert_expected_inline(
            str(mul_summary),
            """{'operator': 'torch.ops.aten.mul.Tensor', 'total_tests': 3, 'correctness_tests': 1, 'performance_tests': 1, 'passed_correctness_tests': 0, 'passed_performance_tests': 1, 'failed_correctness_tests': 1, 'failed_performance_tests': 0, 'correctness_rate': 0.0, 'avg_speedup': 0.0, 'geomean_speedup': 0.0, 'max_absolute_error': -inf, 'max_relative_error': -inf}""",
        )

        sin_summary = op_summaries["torch.ops.aten.sin.default"]
        assert_expected_inline(
            str(sin_summary),
            """{'operator': 'torch.ops.aten.sin.default', 'total_tests': 3, 'correctness_tests': 1, 'performance_tests': 1, 'passed_correctness_tests': 1, 'passed_performance_tests': 0, 'failed_correctness_tests': 0, 'failed_performance_tests': 1, 'correctness_rate': 1.0, 'avg_speedup': 0.0, 'geomean_speedup': 0.0, 'max_absolute_error': 0.0, 'max_relative_error': 0.0}""",
        )

    def test_get_summary_op_results(self):
        """Test the _get_summary_op_results function."""
        correctness_results, performance_results = self._create_test_fixtures()

        op_results = _get_summary_op_results(performance_results, correctness_results)

        # Should return list of tuples (op_name, correctness_str, speedup_str)
        assert len(op_results) == 3

        # Check that results are sorted properly (by speedup descending, then correctness)
        assert_expected_inline(
            str(op_results),
            """[('torch.ops.aten.add.Tensor', '1.0000%', '1.7500x'), ('torch.ops.aten.sin.default', '1.0000%', '1.0000x'), ('torch.ops.aten.mul.Tensor', '0.0000%', '0.0000x')]""",
        )

    def test_save_results_integration(self):
        """Test the full save_results function with file I/O."""
        correctness_results, performance_results = self._create_test_fixtures()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output"

            save_results(
                correctness_results=correctness_results,
                performance_results=performance_results,
                output_path=output_path,
                command="backendbench --suite test_suite",
                mean_correctness=0.75,
                geomean_perf=1.8,
                perf_at_p_score=0.6,
                p=1.2,
            )

            # Check that all expected files were created
            assert (output_path / "full_results.json").exists()
            assert (output_path / "operator_summary.csv").exists()
            assert (output_path / "failed_tests.json").exists()
            assert (output_path / "OVERALL_SUMMARY.md").exists()

            # Check full_results.json content
            with open(output_path / "full_results.json") as f:
                full_results = json.load(f)
            assert len(full_results) == 8

            # Check failed_tests.json content
            with open(output_path / "failed_tests.json") as f:
                failed_tests = json.load(f)
            assert len(failed_tests) == 2

            # Check that CSV has correct number of rows (header + 3 operators)
            with open(output_path / "operator_summary.csv") as f:
                csv_content = f.read()
            # Should have header + 3 data rows
            assert len(csv_content.strip().split("\n")) == 4

            # Check overall summary exists and has expected content
            with open(output_path / "OVERALL_SUMMARY.md") as f:
                summary_content = f.read()
            assert "# BackendBench Run Summary" in summary_content
            assert "backendbench --suite test_suite" in summary_content
            assert "0.75" in summary_content  # mean_correctness
            assert "1.80" in summary_content  # geomean_perf

    def test_save_overall_summary_standalone(self):
        """Test the save_overall_summary function independently."""
        correctness_results, performance_results = self._create_test_fixtures()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_summary"

            save_overall_summary(
                output_path=output_path,
                command="backendbench --ops add,mul",
                mean_correctness=0.8,
                geomean_perf=2.1,
                perf_at_p_score=0.7,
                p=1.5,
                performance_results=performance_results,
                correctness_results=correctness_results,
            )

            # Check that the summary file was created
            summary_path = output_path / "OVERALL_SUMMARY.md"
            assert summary_path.exists()

            # Check content
            with open(summary_path) as f:
                content = f.read()

            assert "backendbench --ops add,mul" in content
            assert "| Correctness Score | 0.80 |" in content
            assert "| Performance Score (geomean speedup) | 2.10 |" in content
            assert "| Perf@1.5 Score | 0.70 |" in content

    def test_empty_results(self):
        """Test functions with empty input data."""
        empty_correctness = []
        empty_performance = []

        all_results, failed_tests, op_summaries = _prepare_results_data(
            empty_correctness, empty_performance
        )

        assert len(all_results) == 0
        assert len(failed_tests) == 0
        assert len(op_summaries) == 0

        # Test with empty results in summary function
        op_results = _get_summary_op_results(empty_performance, empty_correctness)
        assert len(op_results) == 0

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with NaN and infinite values
        edge_case_results = [
            CorrectnessTestResult(
                op_name="edge_case_op",
                args="[tensor([nan])]",
                is_correct=True,
                max_abs_error=float("inf"),
                max_rel_error=-math.inf,
            ),
            PerformanceTestResult(
                op_name="edge_case_op",
                args="[tensor([nan])]",
                speedup=float("inf"),
                benchmark_time_ms=0.0,
                reference_time_ms=1.0,
                successfully_ran=True,
            ),
        ]

        all_results, failed_tests, op_summaries = _prepare_results_data(
            [edge_case_results[0]], [edge_case_results[1]]
        )

        # Should handle infinite values gracefully
        assert len(all_results) == 2
        assert len(op_summaries) == 1

        # Check that infinite speedup is handled
        edge_summary = op_summaries["edge_case_op"]
        assert math.isinf(edge_summary["avg_speedup"])
