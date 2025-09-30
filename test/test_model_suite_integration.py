# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Essential integration tests for Model Suite PR #3: Final Polish

This test suite validates:
1. Complete CLI workflow with model suite
2. Filtering functionality
3. Error handling for invalid backends
4. Operator-level and model-level testing integration
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from BackendBench.suite.model import ModelSuite


class TestModelSuiteCLI(unittest.TestCase):
    """Test CLI integration for model suite."""

    def test_complete_workflow(self):
        """Test complete workflow from CLI."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "BackendBench.scripts.main",
                "--suite",
                "model",
                "--backend",
                "directory",
                "--ops-directory",
                "generated_kernels",
                "--disable-output-logs",
                "--log-level",
                "ERROR",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, "CLI should succeed")
        self.assertIn("FULL MODEL TESTING", result.stdout)
        self.assertIn("Model Suite Score", result.stdout)
        self.assertIn("toy_core_ops", result.stdout)

    def test_filtering_by_model(self):
        """Test filtering models by name."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "BackendBench.scripts.main",
                "--suite",
                "model",
                "--backend",
                "directory",
                "--ops-directory",
                "generated_kernels",
                "--model-filter",
                "toy_core_ops",
                "--disable-output-logs",
                "--log-level",
                "ERROR",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, "Filtered run should succeed")
        self.assertIn("toy_core_ops", result.stdout)

    def test_invalid_backend_error(self):
        """Test that model suite rejects invalid backends."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "BackendBench.scripts.main",
                "--suite",
                "model",
                "--backend",
                "aten",
                "--disable-output-logs",
                "--log-level",
                "ERROR",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertNotEqual(result.returncode, 0, "Should fail with invalid backend")
        self.assertIn("model suite only supports directory backend", result.stderr.lower())

    def test_empty_filter(self):
        """Test handling of nonexistent model filter."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "BackendBench.scripts.main",
                "--suite",
                "model",
                "--backend",
                "directory",
                "--ops-directory",
                "generated_kernels",
                "--model-filter",
                "nonexistent_model",
                "--disable-output-logs",
                "--log-level",
                "ERROR",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should fail because explicitly requested model not found
        self.assertNotEqual(result.returncode, 0, "Should fail with nonexistent filter")

    def test_ops_filter_rejected(self):
        """Test that --ops filter is rejected for model suite."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "BackendBench.scripts.main",
                "--suite",
                "model",
                "--backend",
                "directory",
                "--ops-directory",
                "generated_kernels",
                "--ops",
                "toy_core_ops",
                "--disable-output-logs",
                "--log-level",
                "ERROR",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should fail with error message about --ops not supported
        self.assertNotEqual(result.returncode, 0, "Should fail with --ops")
        self.assertIn("--ops filter is not supported for model suite", result.stderr)


class TestModelSuiteIntegration(unittest.TestCase):
    """Test ModelSuite integration and initialization."""

    def test_initialization_variants(self):
        """Test ModelSuite initialization with various options."""
        # Default initialization
        suite1 = ModelSuite()
        self.assertGreater(len(suite1.models), 0, "Should load models by default")

        # With filter
        suite2 = ModelSuite(filter=["toy_core_ops"])
        self.assertEqual(len(suite2.models), 1, "Should load exactly 1 model")
        self.assertEqual(suite2.models[0]["name"], "toy_core_ops")

        # Empty filter - should raise error
        with self.assertRaises(ValueError) as context:
            _ = ModelSuite(filter=["nonexistent"])
        self.assertIn("No models found", str(context.exception))

    def test_operator_level_integration(self):
        """Test that operator-level testing works via __iter__."""
        suite = ModelSuite()
        op_tests = list(suite)

        # Model suite currently returns empty iterator
        # Operator extraction from model tracing is not yet implemented
        # The suite focuses on full model testing via test_model_correctness()
        self.assertEqual(len(op_tests), 0, "Operator extraction not yet implemented")

    def test_model_level_integration(self):
        """Test that model-level testing works."""
        suite = ModelSuite()
        results = suite.test_model_correctness()

        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0, "Should have results")

        # Verify structure
        for model_name, config_results in results.items():
            self.assertIsInstance(config_results, dict)
            for config_name, is_correct in config_results.items():
                self.assertIsInstance(is_correct, bool)

    def test_output_format(self):
        """Test that CLI output is properly formatted."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "BackendBench.scripts.main",
                "--suite",
                "model",
                "--backend",
                "directory",
                "--ops-directory",
                "generated_kernels",
                "--disable-output-logs",
                "--log-level",
                "ERROR",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = result.stdout

        # Check for expected sections
        self.assertIn("correctness score", output.lower())
        self.assertIn("performance score", output.lower())
        self.assertIn("FULL MODEL TESTING", output)
        self.assertIn("Model Correctness Results:", output)
        self.assertIn("Model Suite Score:", output)

        # Check for formatting
        self.assertIn("=" * 80, output)
        self.assertIn("-" * 80, output)

        # Check for pass/fail indicators
        has_pass_fail = "✓ PASS" in output or "✗ FAIL" in output
        self.assertTrue(has_pass_fail, "Should show pass/fail indicators")


if __name__ == "__main__":
    unittest.main(verbosity=2)
