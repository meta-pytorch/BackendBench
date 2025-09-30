# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Essential tests for Model Suite PR #2: Full Model Testing & Results

This test suite validates the core functionality:
1. FullModelTest class with eager/backend execution
2. Numerical correctness comparison
3. ModelSuite.test_model_correctness() integration
"""

import logging
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from BackendBench.suite.model import FullModelTest, load_toy_models, ModelSuite

# Setup logging
logging.basicConfig(level=logging.WARNING)


class TestFullModelTest(unittest.TestCase):
    """Test FullModelTest class functionality."""

    @classmethod
    def setUpClass(cls):
        """Load toy models once for all tests."""
        cls.models = load_toy_models(toy_models_dir="BackendBench/suite/models")
        assert len(cls.models) > 0, "Should load at least one model"
        cls.model = next(m for m in cls.models if m["name"] == "toy_core_ops")

    def test_initialization(self):
        """Test FullModelTest can be instantiated correctly."""
        test_name = list(self.model["config"]["model_tests"].keys())[0]
        test_args = self.model["config"]["model_tests"][test_name]
        full_test = FullModelTest(
            model_name=self.model["name"],
            model_class=self.model["class"],
            model_config=self.model["config"]["model_config"],
            test_name=test_name,
            test_args=test_args,
        )

        self.assertEqual(full_test.model_name, self.model["name"])
        self.assertEqual(full_test.model_class, self.model["class"])

    def test_eager_execution(self):
        """Test model runs correctly in eager mode."""
        test_name = "small_batch"
        test_args = self.model["config"]["model_tests"][test_name]
        full_test = FullModelTest(
            self.model["name"],
            self.model["class"],
            self.model["config"]["model_config"],
            test_name,
            test_args,
        )

        output, grads = full_test.run_with_backend(backend_enabled=False)

        # Verify output shape (batch_size=2 from small_batch config)
        expected_shape = torch.Size([2, 8, 4, 4])
        self.assertEqual(output.shape, expected_shape)

        # Verify gradients computed
        self.assertGreater(len(grads), 0, "Should compute gradients")

        # Verify all gradients are valid
        for grad in grads:
            self.assertIsInstance(grad, torch.Tensor)
            self.assertFalse(torch.isnan(grad).any(), "No NaN gradients")
            self.assertFalse(torch.isinf(grad).any(), "No Inf gradients")

    def test_backend_execution(self):
        """Test model runs with backend enabled."""
        test_name = "small_batch"
        test_args = self.model["config"]["model_tests"][test_name]
        full_test = FullModelTest(
            self.model["name"],
            self.model["class"],
            self.model["config"]["model_config"],
            test_name,
            test_args,
        )

        output, grads = full_test.run_with_backend(backend_enabled=True)

        # Verify output shape (batch_size=2 from small_batch config)
        expected_shape = torch.Size([2, 8, 4, 4])
        self.assertEqual(output.shape, expected_shape)

        # Verify gradients computed
        self.assertGreater(len(grads), 0, "Should compute gradients")

    def test_correctness_comparison(self):
        """Test correctness comparison between eager and backend."""
        test_name = "small_batch"
        test_args = self.model["config"]["model_tests"][test_name]
        full_test = FullModelTest(
            self.model["name"],
            self.model["class"],
            self.model["config"]["model_config"],
            test_name,
            test_args,
        )

        is_correct = full_test.test_correctness()

        # Result should be a boolean
        self.assertIsInstance(is_correct, bool)

        # With existing kernels, test should pass
        self.assertTrue(is_correct, "Backend should produce correct results")

    def test_multiple_configs(self):
        """Test all model configurations run correctly."""
        # Expected shapes for each config
        # Note: Output size is always 4x4 due to adaptive_avg_pool2d([4, 4])
        expected_shapes = {
            "small_batch": torch.Size([2, 8, 4, 4]),
            "medium_batch": torch.Size([4, 8, 4, 4]),
            "large_input": torch.Size([2, 8, 4, 4]),
        }

        for test_name, test_args in self.model["config"]["model_tests"].items():
            full_test = FullModelTest(
                self.model["name"],
                self.model["class"],
                self.model["config"]["model_config"],
                test_name,
                test_args,
            )

            output, grads = full_test.run_with_backend(backend_enabled=False)

            expected_shape = expected_shapes[test_name]
            self.assertEqual(output.shape, expected_shape, f"Config {test_name} failed")
            self.assertGreater(len(grads), 0, f"Config {test_name} has no gradients")


class TestModelSuite(unittest.TestCase):
    """Test ModelSuite.test_model_correctness() integration."""

    def test_model_correctness_method_exists(self):
        """Test that test_model_correctness method exists."""
        suite = ModelSuite()
        self.assertTrue(hasattr(suite, "test_model_correctness"))

    def test_model_correctness_integration(self):
        """Test ModelSuite.test_model_correctness() returns proper results."""
        suite = ModelSuite()
        results = suite.test_model_correctness()

        # Verify results structure
        self.assertIsInstance(results, dict, "Results should be a dictionary")
        self.assertGreater(len(results), 0, "Should have results for at least one model")

        # Verify each model has config results
        for model_name, model_results in results.items():
            self.assertIsInstance(model_results, dict, f"{model_name} results should be dict")
            self.assertGreater(len(model_results), 0, f"{model_name} should have test configs")

            # Verify each config result is a boolean
            for config_name, is_correct in model_results.items():
                self.assertIsInstance(
                    is_correct, bool, f"{model_name}::{config_name} should be bool"
                )

    def test_results_aggregation(self):
        """Test that results can be aggregated for scoring."""
        suite = ModelSuite()
        results = suite.test_model_correctness()

        # Calculate aggregate statistics
        total_tests = sum(len(model_results) for model_results in results.values())
        total_passed = sum(
            sum(1 for result in model_results.values() if result)
            for model_results in results.values()
        )

        self.assertGreater(total_tests, 0, "Should have at least one test")
        self.assertLessEqual(total_passed, total_tests, "Passed <= Total")
        self.assertGreaterEqual(total_passed, 0, "Passed >= 0")

    def test_empty_filter(self):
        """Test suite raises error for nonexistent model."""
        with self.assertRaises(ValueError) as context:
            _ = ModelSuite(filter=["nonexistent_model"])
        self.assertIn("No models found", str(context.exception))


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
