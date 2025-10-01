# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit test to verify that models actually invoke all operators declared in their configs.

This test validates that:
1. Forward pass invokes all operators in config["ops"]["forward"]
2. Backward pass invokes all operators in config["ops"]["backward"]
3. Clear error messages indicate which operators are missing per model
"""

import os
import re
import sys
import unittest
from typing import Dict, Set

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from BackendBench.suite.model import load_models


class OpTracker:
    """Track operators called during forward/backward passes using torch profiler."""

    def __init__(self):
        self.called_ops: Set[str] = set()
        self.profiler = None

    def __enter__(self):
        self.called_ops.clear()

        # Use torch profiler to track ops
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
            with_stack=False,
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, *args):
        self.profiler.__exit__(*args)

        # Extract op names from profiler events
        for event in self.profiler.events():
            event_name = event.name
            # Look for aten operations
            if "::" in event_name:
                # Handle format like "aten::convolution" or "aten::convolution.default"
                parts = event_name.replace("::", ".").split(".")

                if len(parts) >= 2 and parts[0] == "aten":
                    if len(parts) == 2:
                        # No variant specified, add .default
                        op_name = f"{parts[0]}.{parts[1]}.default"
                    else:
                        # Keep as is
                        op_name = event_name.replace("::", ".")

                    self.called_ops.add(op_name)


class TestModelOpsCoverage(unittest.TestCase):
    """Test that models invoke all operators declared in their configs."""

    def test_all_models_ops_coverage(self):
        """Test that all models invoke their declared forward and backward ops."""
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "BackendBench",
            "suite",
            "models",
        )

        models = load_models(models_dir=models_dir)
        self.assertGreater(len(models), 0, "Should load at least one model")

        failures = []

        for model_dict in models:
            model_name = model_dict["name"]
            model_class = model_dict["class"]
            config = model_dict["config"]

            # Get expected ops from config
            config_ops = config.get("ops", {})
            if isinstance(config_ops, list):
                # Legacy format - skip or treat as forward-only
                expected_forward = set(config_ops)
                expected_backward = set()
            else:
                expected_forward = set(config_ops.get("forward", []))
                expected_backward = set(config_ops.get("backward", []))

            # Skip if no ops to check
            if not expected_forward and not expected_backward:
                continue

            try:
                # Initialize model
                model_config = config.get("model_config", {})
                init_args = model_config.get("init_args", {})

                if model_config.get("requires_init_seed"):
                    torch.manual_seed(42)

                model = model_class(**init_args)

                # Get a test input from model_tests
                model_tests = config.get("model_tests", {})
                if not model_tests:
                    failures.append(f"{model_name}: No model_tests in config")
                    continue

                # Use first test case
                test_name = list(model_tests.keys())[0]
                test_args_str = model_tests[test_name]

                # Parse test args (simple eval for now)
                # Format: "([], {'x': T([2, 3, 32, 32], f32)})"
                test_input = self._create_test_input_from_string(test_args_str)

                # Track forward pass
                tracker = OpTracker()
                with tracker:
                    output = model(**test_input)

                forward_ops = tracker.called_ops

                # Check forward ops coverage
                missing_forward = expected_forward - forward_ops
                if missing_forward:
                    failures.append(
                        f"{model_name} [FORWARD]: Missing ops: {sorted(missing_forward)}"
                    )

                # Track backward pass
                if expected_backward:
                    # Ensure output requires grad
                    for param in model.parameters():
                        param.requires_grad = True

                    # Create loss
                    if isinstance(output, torch.Tensor):
                        loss = output.sum()
                    else:
                        # Handle tuple/dict outputs
                        loss = sum(v.sum() for v in output.values() if isinstance(v, torch.Tensor))

                    tracker_backward = OpTracker()
                    with tracker_backward:
                        loss.backward()

                    backward_ops = tracker_backward.called_ops

                    # Check backward ops coverage
                    missing_backward = expected_backward - backward_ops
                    if missing_backward:
                        failures.append(
                            f"{model_name} [BACKWARD]: Missing ops: {sorted(missing_backward)}"
                        )

            except Exception as e:
                failures.append(f"{model_name}: Error during test: {e}")

        # Report all failures at once
        if failures:
            error_msg = "\n\nOperator Coverage Failures:\n" + "\n".join(
                f"  - {failure}" for failure in failures
            )
            self.fail(error_msg)

    def _create_test_input_from_string(self, test_args_str: str) -> Dict[str, torch.Tensor]:
        """Parse test input string into actual tensors.

        Format: "([], {'x': T([2, 3, 32, 32], f32)})"
        """

        # Extract tensor specs: T([shape], dtype)
        tensor_pattern = r"'(\w+)':\s*T\(\[([\d,\s]+)\],\s*(\w+)\)"
        matches = re.findall(tensor_pattern, test_args_str)

        inputs = {}
        for name, shape_str, dtype_str in matches:
            shape = [int(x.strip()) for x in shape_str.split(",")]

            # Map dtype string to torch dtype
            dtype_map = {
                "f32": torch.float32,
                "f64": torch.float64,
                "i32": torch.int32,
                "i64": torch.int64,
            }
            dtype = dtype_map.get(dtype_str, torch.float32)

            inputs[name] = torch.randn(*shape, dtype=dtype)

        return inputs


if __name__ == "__main__":
    unittest.main(verbosity=2)
