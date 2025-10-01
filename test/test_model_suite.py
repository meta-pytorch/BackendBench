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
import unittest

from BackendBench.suite.model import load_models

# Setup logging
logging.basicConfig(level=logging.WARNING)


class TestModelLoading(unittest.TestCase):
    """Test model loading functionality."""

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


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
