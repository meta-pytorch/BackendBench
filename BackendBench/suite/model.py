# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Model Suite for testing operators defined in toy model configs.

This suite extends TorchBenchTestSuite by reading operator lists from
model configs, validating they exist in the TorchBench dataset, then
filtering to include only those operators.
"""

import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from BackendBench.eval_model import (
    eval_model_correctness_test,
    eval_model_performance_test,
)

from .torchbench import TorchBenchTestSuite

logger = logging.getLogger(__name__)


def load_models(
    models_dir: str = "models", filter: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Load models using strict naming convention: folder_name/folder_name.py + folder_name.json

    Args:
        models_dir: Directory containing models (default: "models")
        filter: Optional list of model names to load. If None, loads all models.

    Returns:
        List of dictionaries with keys:
        - name: Model name (str)
        - class: Model class (type)
        - config: Configuration dictionary from JSON file
    """
    models = []

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    for model_name in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # Skip if not in filter
        if filter is not None and model_name not in filter:
            continue

        # Strict naming convention: folder_name/folder_name.py and folder_name/folder_name.json
        model_file = os.path.join(model_dir, f"{model_name}.py")
        config_file = os.path.join(model_dir, f"{model_name}.json")

        # Check both files exist
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        try:
            # Load config
            with open(config_file, "r") as f:
                config = json.load(f)

            # Load model class dynamically
            spec = importlib.util.spec_from_file_location(model_name, model_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find model class (must match model_name exactly)
            if not hasattr(module, model_name):
                raise RuntimeError(f"Model class '{model_name}' not found in {model_file}")

            model_class = getattr(module, model_name)
            if not (isinstance(model_class, type) and hasattr(model_class, "forward")):
                raise RuntimeError(f"'{model_name}' in {model_file} is not a valid model class")

            models.append({"name": model_name, "class": model_class, "config": config})
            logger.info(f"Loaded model: {model_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    if filter is not None and len(models) == 0:
        raise ValueError(f"No models found matching filter: {filter}")

    return models


class ModelSuite(TorchBenchTestSuite):
    """Model Suite that filters TorchBench operators based on model configs.

    This suite reads operator lists from model configs, validates they exist
    in the TorchBench dataset, then creates a filtered suite containing only
    those operators.
    """

    def __init__(
        self,
        name: str = "model",
        filter: Optional[List[str]] = None,
        topn: Optional[int] = None,
    ):
        """Initialize ModelSuite.

        Args:
            name: Suite name (default: "model")
            filter: Optional list of model names to load
            topn: Optional limit on number of tests per operator
        """
        models_dir = os.path.join(os.path.dirname(__file__), "models")

        # Load models
        models = load_models(models_dir=models_dir, filter=filter)
        logger.info(f"ModelSuite: Loaded {len(models)} models from {models_dir}")
        model_ops = self.get_model_ops(models)
        filter = list(model_ops)
        # Store loaded models for evaluation
        self.models = models

        self._initialize_torchbench_suite(name, None, filter, topn, False)

    def get_model_ops(self, models: List[Dict[str, Any]]) -> List[str]:
        # Extract operators from model configs
        model_ops = set()
        for model in models:
            config_ops = model["config"]["ops"]
            ops_list = config_ops["forward"]
            ops_list.extend(config_ops["backward"])

            model_ops.update(ops_list)
            logger.info(f"Model {model['name']}: {len(ops_list)} operators defined in config")

        logger.info(f"ModelSuite: Total {len(model_ops)} unique operators across all models")
        return model_ops

    def eval_model(self, model_dict: Dict[str, Any], backend) -> Dict[str, Any]:
        """Run evaluation on a single model.

        Args:
            model_dict: Dictionary with keys 'name', 'class', 'config'
            backend: Backend to use for evaluation

        Returns:
            Dictionary with evaluation results including correctness and performance
        """

        model_class = model_dict["class"]
        model_name = model_dict["name"]
        config = model_dict["config"]

        # Extract model configuration and tests
        model_config = config.get("model_config", {})
        model_tests = config.get("model_tests", {})

        if not model_tests:
            return {
                "model_name": model_name,
                "passed": False,
                "error": "No model_tests found in config",
                "correctness_results": [],
                "performance_results": [],
            }

        # Get kernel_dir from backend if available
        kernel_dir = getattr(backend, "ops_dir", None)

        # Run both correctness and performance tests
        correctness_results = []
        performance_results = []

        for test_name, test_args in model_tests.items():
            # Correctness test
            corr_result = eval_model_correctness_test(
                model_name=model_name,
                model_class=model_class,
                model_config=model_config,
                test_name=test_name,
                test_args=test_args,
                kernel_dir=kernel_dir,
            )
            correctness_results.append(corr_result)

            # Performance test
            perf_result = eval_model_performance_test(
                model_name=model_name,
                model_class=model_class,
                model_config=model_config,
                test_name=test_name,
                test_args=test_args,
                kernel_dir=kernel_dir,
            )
            performance_results.append(perf_result)

        # Aggregate results
        all_passed = all(r.is_correct for r in correctness_results)
        num_passed = sum(1 for r in correctness_results if r.is_correct)
        num_total = len(correctness_results)

        # Calculate average speedup (geometric mean like operators)
        speedups = [r.speedup for r in performance_results if r.successfully_ran]
        avg_speedup = torch.tensor(speedups).log().mean().exp().item() if speedups else 0.0

        return {
            "model_name": model_name,
            "passed": all_passed,
            "num_passed": num_passed,
            "num_total": num_total,
            "correctness_results": correctness_results,
            "performance_results": performance_results,
            "avg_speedup": avg_speedup,
        }

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print model evaluation results.

        Args:
            results: Dictionary with evaluation results from eval_model
        """
        model_name = results.get("model_name", "Unknown")
        passed = results.get("passed", False)
        num_passed = results.get("num_passed", 0)
        num_total = results.get("num_total", 0)
        avg_speedup = results.get("avg_speedup", 0.0)

        logger.info(f"\nModel: {model_name}")
        logger.info(
            f"Status: {'✓ Passed' if passed else '✗ Failed'} ({num_passed}/{num_total} tests)"
        )
        logger.info(f"Performance: {avg_speedup:.2f}x average speedup")

        # Print details for each test
        correctness_results = results.get("correctness_results", [])
        performance_results = results.get("performance_results", [])

        # Handle backward compatibility
        if not correctness_results:
            correctness_results = results.get("test_results", [])

        for corr_result in correctness_results:
            # Find corresponding performance result
            perf_result = None
            for p in performance_results:
                if p.test_name == corr_result.test_name:
                    perf_result = p
                    break

            status = "✓" if corr_result.is_correct else "✗"
            speedup_str = ""
            if perf_result and perf_result.successfully_ran:
                speedup_str = f" ({perf_result.speedup:.2f}x speedup)"

            logger.info(f"  {status} {corr_result.test_name}{speedup_str}")

            if not corr_result.is_correct:
                if corr_result.error_msg:
                    logger.info(f"    Error: {corr_result.error_msg}")
                else:
                    # Show what failed
                    if not corr_result.output_match:
                        logger.info("    Output mismatch")
                    if not corr_result.gradients_match:
                        logger.info(
                            f"    Gradient mismatch ({corr_result.num_gradients} gradients)"
                        )
            else:
                # Show success details
                details = f"    Output match: ✓  Gradients match: ✓ ({corr_result.num_gradients} gradients)"
                if perf_result and perf_result.successfully_ran:
                    details += f"  Time: eager={perf_result.eager_time_ms:.2f}ms, backend={perf_result.backend_time_ms:.2f}ms"
                logger.info(details)
