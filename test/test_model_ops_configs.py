# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit test to verify that ModelSuite's operator filter correctly matches
the operators defined in model configs.

This test validates that:
1. load_models correctly loads model configs from the models directory
2. load_model_ops extracts the correct set of operators from model configs
3. TorchBenchTestSuite initialized with those operators has matching optests
4. JSON config files have proper format with required fields
"""

import json
import os
import unittest
from typing import Any, Dict, List, Set

from BackendBench.suite.model import load_models
from BackendBench.suite.torchbench import TorchBenchTestSuite


def load_model_ops(models: List[Dict[str, Any]]) -> Set[str]:
    """Extract unique set of operators from model configs.

    Args:
        models: List of model dictionaries with 'name', 'class', and 'config' keys

    Returns:
        Set of operator names defined across all model configs
    """
    model_ops = set()
    for model in models:
        config_ops = model["config"].get("ops")
        if not config_ops:
            raise ValueError(f"Model {model['name']} has no 'ops' field in config")
        assert "forward" in config_ops, f"Model {model['name']} has no 'forward' field in config"
        assert "backward" in config_ops, f"Model {model['name']} has no 'backward' field in config"
        ops_list = config_ops["forward"] + config_ops["backward"]

        model_ops.update(ops_list)
    return model_ops


class TestModelOpsConfigs(unittest.TestCase):
    """Test that model ops filter correctly initializes TorchBenchTestSuite."""

    def test_model_ops_match_suite_optests(self):
        """Test that suite's optests match the operators from model configs."""
        # Get the models directory path (same as ModelSuite does)
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "BackendBench", "suite", "models"
        )

        # Load models using load_models
        models = load_models(models_dir=models_dir)

        # Verify we loaded at least one model
        self.assertGreater(len(models), 0, "Should load at least one model")

        # Extract operators from model configs using load_model_ops
        model_ops = load_model_ops(models)

        # Verify we have operators
        self.assertGreater(len(model_ops), 0, "Should have at least one operator")

        # Create filter list from model ops
        ops_filter = list(model_ops)

        # Initialize TorchBenchTestSuite with the filter
        suite = TorchBenchTestSuite(
            name="test_model_ops",
            filename=None,  # Use default HuggingFace dataset
            filter=ops_filter,
            topn=None,
        )

        # Get the set of operators in the suite's optests
        suite_ops = set(suite.optests.keys())

        # The suite_ops should be a subset of model_ops because:
        # - model_ops is the filter we requested
        # - suite_ops contains only those operators that exist in the TorchBench dataset
        # - Not all operators in model configs may be in the dataset
        self.assertTrue(
            suite_ops.issubset(model_ops),
            f"Suite operators {suite_ops} should be subset of model operators {model_ops}",
        )

        # Verify that suite actually has some operators
        self.assertGreater(
            len(suite_ops), 0, "Suite should contain at least one operator from model configs"
        )

    def test_json_configs_have_required_fields(self):
        """Test that all JSON config files have proper format with required fields."""
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "BackendBench", "suite", "models"
        )

        # Load all models
        models = load_models(models_dir=models_dir)

        for model in models:
            model_name = model["name"]
            config = model["config"]

            # Check required top-level fields
            self.assertIn("ops", config, f"Model {model_name}: config must have 'ops' field")
            self.assertIn(
                "model_tests", config, f"Model {model_name}: config must have 'model_tests' field"
            )

            # Validate 'ops' field - can be list or dict
            config_ops = config["ops"]
            self.assertGreater(
                len(config_ops["forward"] + config_ops["backward"]),
                0,
                f"Model {model_name}: 'ops' list must not be empty",
            )
            for op in config_ops["forward"] + config_ops["backward"]:
                self.assertIsInstance(
                    op, str, f"Model {model_name}: each op in 'ops' must be a string"
                )
            self.assertIsInstance(
                config_ops["forward"],
                list,
                f"Model {model_name}: 'ops.forward' must be a list",
            )
            for op in config_ops["forward"]:
                self.assertIsInstance(
                    op,
                    str,
                    f"Model {model_name}: each op in 'ops.forward' must be a string",
                )
            self.assertIsInstance(
                config_ops["backward"],
                list,
                f"Model {model_name}: 'ops.backward' must be a list",
            )
            for op in config_ops["backward"]:
                self.assertIsInstance(
                    op,
                    str,
                    f"Model {model_name}: each op in 'ops.backward' must be a string",
                )

            # Validate 'model_tests' field
            self.assertIsInstance(
                config["model_tests"],
                dict,
                f"Model {model_name}: 'model_tests' must be a dictionary",
            )
            self.assertGreater(
                len(config["model_tests"]),
                0,
                f"Model {model_name}: 'model_tests' must not be empty",
            )

            # Validate 'model_tests' field
            self.assertIsInstance(
                config["model_tests"],
                dict,
                f"Model {model_name}: 'model_tests' must be a dictionary",
            )
            self.assertGreater(
                len(config["model_tests"]),
                0,
                f"Model {model_name}: 'model_tests' must not be empty",
            )
            for test_name, test_args in config["model_tests"].items():
                self.assertIsInstance(
                    test_name, str, f"Model {model_name}: test names must be strings"
                )
                self.assertIsInstance(
                    test_args, str, f"Model {model_name}: test args must be strings"
                )

            # Check optional but recommended fields
            if "model_config" in config:
                self.assertIsInstance(
                    config["model_config"],
                    dict,
                    f"Model {model_name}: 'model_config' must be a dictionary if present",
                )

    def test_json_files_are_valid_json(self):
        """Test that all JSON config files are valid JSON and can be parsed."""
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "BackendBench", "suite", "models"
        )

        # Find all JSON files in the models directory
        for model_name in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            json_file = os.path.join(model_dir, f"{model_name}.json")
            if not os.path.exists(json_file):
                continue

            # Try to parse the JSON file
            with open(json_file, "r") as f:
                try:
                    config = json.load(f)
                    self.assertIsInstance(
                        config,
                        dict,
                        f"JSON file {json_file} must contain a dictionary at top level",
                    )
                except json.JSONDecodeError as e:
                    self.fail(f"JSON file {json_file} is not valid JSON: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
