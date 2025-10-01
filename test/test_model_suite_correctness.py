# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Model Suite: Filtered TorchBench operators from model tracing

This test suite validates:
1. Model loading from toy_models directory
2. Operator extraction via model tracing
3. ModelSuite creates filtered TorchBench suite
"""

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from BackendBench.suite.model import load_models, ModelSuite

# Setup logging
logging.basicConfig(level=logging.WARNING)


class TestModelLoading(unittest.TestCase):
    """Test toy model loading functionality."""

    def test_load_models(self):
        """Test that models can be loaded from directory."""
        models = load_models(models_dir="BackendBench/suite/models")
        self.assertGreater(len(models), 0, "Should load at least one model")

        # Verify model structure
        for model in models:
            self.assertIn("name", model)
            self.assertIn("class", model)
            self.assertIn("config", model)

    def test_load_specific_model(self):
        """Test loading a specific model by name."""
        models = load_models(models_dir="BackendBench/suite/models", filter=["ToyCoreOpsModel"])
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "ToyCoreOpsModel")

    def test_invalid_filter(self):
        """Test that invalid filter raises error."""
        with self.assertRaises(ValueError):
            load_models(models_dir="BackendBench/suite/models", filter=["nonexistent"])


class TestModelSuite(unittest.TestCase):
    """Test ModelSuite integration with TorchBench."""

    def test_suite_initialization(self):
        """Test that ModelSuite can be initialized."""
        suite = ModelSuite()
        self.assertEqual(suite.name, "model")
        self.assertIsNotNone(suite.optests)

    def test_suite_has_operators(self):
        """Test that suite extracts operators from models."""
        suite = ModelSuite()
        # Should have extracted and filtered operators
        self.assertGreater(len(suite.optests), 0, "Should have at least one operator")

    def test_suite_iteration(self):
        """Test that suite can be iterated (TorchBench interface)."""
        suite = ModelSuite()
        op_tests = list(suite)
        # Should have at least one operator test
        self.assertGreater(len(op_tests), 0, "Should have at least one operator test")

    def test_empty_filter(self):
        """Test suite raises error for nonexistent model."""
        with self.assertRaises(ValueError):
            _ = ModelSuite(filter=["nonexistent_model"])


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
