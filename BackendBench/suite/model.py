# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Model Suite for testing toy models against backends.

This suite extends TorchBenchTestSuite to provide two testing approaches:
1. Operator-level testing (via __iter__, inherited infrastructure)
2. Model-level correctness testing (via test_model_correctness, new functionality)
"""

import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from BackendBench.data_loaders import load_ops_from_source, op_list_to_benchmark_dict
from BackendBench.utils import deserialize_args

from .torchbench import TorchBenchOpTest, TorchBenchTestSuite

logger = logging.getLogger(__name__)

# Cache for torchbench ops to avoid reloading
_TORCHBENCH_OPS_CACHE = None


def _get_torchbench_ops():
    """Get list of available ops from torchbench dataset (cached)."""
    global _TORCHBENCH_OPS_CACHE
    if _TORCHBENCH_OPS_CACHE is None:
        ops_list = load_ops_from_source(source=None, format="parquet")
        _TORCHBENCH_OPS_CACHE = set(op_list_to_benchmark_dict(ops_list).keys())
    return _TORCHBENCH_OPS_CACHE


def load_toy_models(
    toy_models_dir: str = "models", filter: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Load models using strict naming convention: folder_name/folder_name.py + folder_name.json

    Args:
        toy_models_dir: Directory containing toy models (default: "models")
        filter: Optional list of model names to load. If None, loads all models.

    Returns:
        List of dictionaries with keys:
        - name: Model name (str)
        - class: Model class (type)
        - config: Configuration dictionary from JSON file
    """
    models = []

    if not os.path.exists(toy_models_dir):
        raise FileNotFoundError(f"Toy models directory not found: {toy_models_dir}")

    for model_name in os.listdir(toy_models_dir):
        # Apply filter if specified
        if filter is not None and model_name not in filter:
            continue

        model_dir = os.path.join(toy_models_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # Strict naming convention: folder_name/folder_name.py and folder_name/folder_name.json
        model_file = os.path.join(model_dir, f"{model_name}.py")
        config_file = os.path.join(model_dir, f"{model_name}.json")

        # Check both files exist
        if not os.path.exists(model_file):
            if filter is not None and model_name in filter:
                # If the model was explicitly requested but not found, raise an error
                raise FileNotFoundError(f"Model file not found: {model_file}")
            logger.warning(f"Model file not found: {model_file}")
            continue

        if not os.path.exists(config_file):
            if filter is not None and model_name in filter:
                # If the model was explicitly requested but not found, raise an error
                raise FileNotFoundError(f"Config file not found: {config_file}")
            logger.warning(f"Config file not found: {config_file}")
            continue

        try:
            # Load config
            with open(config_file, "r") as f:
                config = json.load(f)

            # Load model class dynamically
            spec = importlib.util.spec_from_file_location(model_name, model_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find model class (ends with "Model")
            model_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and attr_name.endswith("Model")
                    and hasattr(attr, "forward")
                ):
                    model_class = attr
                    break

            if model_class is None:
                logger.error(f"No model class found in {model_file}")
                continue

            # Generate runtime seed if required
            if config.get("model_config", {}).get("requires_init_seed", False):
                import random

                runtime_seed = random.randint(0, 2**31 - 1)
                config["model_config"]["runtime_seed"] = runtime_seed
                logger.debug(f"Generated runtime seed {runtime_seed} for {model_name}")

            models.append({"name": model_name, "class": model_class, "config": config})
            logger.info(f"Loaded model: {model_name}")

        except Exception as e:
            if filter is not None and model_name in filter:
                # If the model was explicitly requested but failed to load, raise an error
                raise RuntimeError(f"Failed to load model {model_name}: {e}")
            logger.error(f"Failed to load {model_name}: {e}")
            continue

    # If a filter was specified but no models were loaded, raise an error
    if filter is not None and len(models) == 0:
        raise ValueError(f"No models found matching filter: {filter}")

    return models


def _create_op_test(op_name: str, inputs: List[str]):
    """Create a TorchBenchOpTest for a specific operator.

    Args:
        op_name: Operator name in aten format (e.g., "aten.conv2d.default")
        inputs: List of serialized input strings

    Returns:
        TorchBenchOpTest instance compatible with existing evaluation infrastructure

    Raises:
        ValueError: If the op is not in the torchbench dataset
    """
    # Check that the op is in the torchbench dataset
    torchbench_ops = _get_torchbench_ops()
    if op_name not in torchbench_ops:
        raise ValueError(
            f"Operator {op_name} is not in the torchbench dataset. "
            f"Only ops from the torchbench dataset can be tested."
        )

    # Create the test with serialized inputs
    return TorchBenchOpTest(op_name, inputs, topn=None)


class FullModelTest:
    """Complete model forward/backward testing.

    This class handles running a model with a specific test configuration
    in both eager mode and backend mode, then comparing the results.
    """

    def __init__(
        self,
        model_name: str,
        model_class,
        model_config: Dict[str, Any],
        test_name: str,
        test_args: str,
    ):
        """Initialize FullModelTest.

        Args:
            model_name: Name of the model being tested
            model_class: Model class to instantiate
            model_config: Model configuration dict with init_args
            test_name: Name of this test configuration
            test_args: Serialized arguments string for forward pass
        """
        self.model_name = model_name
        self.model_class = model_class
        self.model_config = model_config
        self.test_name = test_name
        self.test_args = test_args

    def run_with_backend(self, backend_enabled: bool, kernel_dir: str = None) -> tuple:
        """Run model with backend enabled or disabled.

        Args:
            backend_enabled: If True, use BackendBench context manager to enable backend
            kernel_dir: Optional directory containing kernels (for backend mode)

        Returns:
            Tuple of (output, gradients) where:
            - output: Model output tensor (detached)
            - gradients: List of gradient tensors [input_grad, param1_grad, param2_grad, ...]
        """
        import BackendBench

        # Deserialize test arguments
        args, kwargs = deserialize_args(self.test_args)

        # Extract model initialization args
        init_args = self.model_config.get("init_args", {}).copy()

        # Handle seed: use runtime_seed if required, otherwise use seed from init_args
        if self.model_config.get("requires_init_seed", False):
            # Use the generated runtime seed
            seed = self.model_config["runtime_seed"]
            init_args["seed"] = seed
        else:
            # Use seed from init_args or default
            seed = init_args.get("seed", 42)

        # Set seed for deterministic behavior
        torch.manual_seed(seed)

        # Create fresh model instance
        model = self.model_class(**init_args)
        model.train()

        # Move model to same device as input (typically CUDA)
        # Check both args and kwargs for tensor
        input_tensor = None
        if args and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
        elif "x" in kwargs and isinstance(kwargs["x"], torch.Tensor):
            input_tensor = kwargs["x"]

        if input_tensor is not None:
            device = input_tensor.device
            model = model.to(device)

        # Ensure input has requires_grad for gradient computation
        if args and isinstance(args[0], torch.Tensor):
            x = args[0]
            if not x.requires_grad:
                x = x.clone().detach().requires_grad_(True)
                args = [x] + list(args[1:])
        elif "x" in kwargs and isinstance(kwargs["x"], torch.Tensor):
            x = kwargs["x"]
            if not x.requires_grad:
                x = x.clone().detach().requires_grad_(True)
                kwargs["x"] = x

        # Run forward + backward with or without backend
        if backend_enabled:
            # Use context manager to enable backend
            if kernel_dir is None:
                # Default to generated_kernels directory
                kernel_dir = os.path.join(os.getcwd(), "generated_kernels")

            with BackendBench.BackendBench.enable(kernel_dir=kernel_dir):
                output = model(*args, **kwargs)
                loss = output.sum()
                loss.backward()
        else:
            # Run in eager mode (no backend)
            output = model(*args, **kwargs)
            loss = output.sum()
            loss.backward()

        # Collect gradients: [input_grad, param1_grad, param2_grad, ...]
        grads = []

        # Input gradient - check both args and kwargs
        input_grad = None
        if args and isinstance(args[0], torch.Tensor) and args[0].grad is not None:
            input_grad = args[0].grad
        elif (
            "x" in kwargs and isinstance(kwargs["x"], torch.Tensor) and kwargs["x"].grad is not None
        ):
            input_grad = kwargs["x"].grad

        if input_grad is not None:
            grads.append(input_grad.clone())

        # Parameter gradients
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.clone())

        return output.detach(), grads

    def test_correctness(self, atol=1e-6, rtol=1e-5, kernel_dir: str = None) -> bool:
        """Test numerical correctness by comparing eager vs backend execution.

        Args:
            atol: Absolute tolerance for torch.allclose
            rtol: Relative tolerance for torch.allclose
            kernel_dir: Optional directory containing kernels

        Returns:
            True if eager and backend produce matching results, False otherwise
        """
        try:
            # Run in eager mode
            eager_out, eager_grads = self.run_with_backend(False, kernel_dir=kernel_dir)

            # Run with backend
            backend_out, backend_grads = self.run_with_backend(True, kernel_dir=kernel_dir)

            # Compare outputs
            if not torch.allclose(eager_out, backend_out, atol=atol, rtol=rtol):
                logger.debug(f"{self.model_name}::{self.test_name}: Output mismatch")
                return False

            # Compare number of gradients
            if len(eager_grads) != len(backend_grads):
                logger.debug(
                    f"{self.model_name}::{self.test_name}: "
                    f"Gradient count mismatch ({len(eager_grads)} vs {len(backend_grads)})"
                )
                return False

            # Compare each gradient
            for i, (eager_grad, backend_grad) in enumerate(zip(eager_grads, backend_grads)):
                if not torch.allclose(eager_grad, backend_grad, atol=atol, rtol=rtol):
                    logger.debug(f"{self.model_name}::{self.test_name}: Gradient {i} mismatch")
                    return False

            return True

        except Exception as e:
            logger.error(f"{self.model_name}::{self.test_name}: Correctness test failed: {e}")
            return False


class ModelSuite(TorchBenchTestSuite):
    """Model Suite extending TorchBenchTestSuite.

    Provides two testing approaches:
    1. Operator-level testing via __iter__() (inherited infrastructure)
    2. Model-level correctness testing via test_model_correctness() (Model Suite specific)
    """

    def __init__(
        self, name: str = "model", filter: Optional[List[str]] = None, models_dir: str = None
    ):
        """Initialize ModelSuite.

        Args:
            name: Suite name (default: "model")
            filter: Optional list of model names to test
            models_dir: Optional directory for models (default: "BackendBench/suite/models")
        """
        # Don't call super().__init__() with parameters since TorchBenchTestSuite
        # expects different arguments. Just initialize the base object.
        super(TorchBenchTestSuite, self).__init__()

        self.name = name

        # Default to models under suite/models
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")

        self.models = load_toy_models(toy_models_dir=models_dir, filter=filter)
        logger.info(f"ModelSuite: {len(self.models)} models loaded from {models_dir}")

    def __iter__(self):
        """Yield operator tests from all models (TorchBench approach).

        This method extracts operators by tracing model execution, then creates
        TorchBenchOpTest instances for testing via the inherited infrastructure.
        """
        # Trace each model to extract operators

        # For now, return empty iterator since operator extraction from model tracing
        # is not yet implemented. The model suite focuses on full model testing via
        # test_model_correctness() method.
        return iter([])

    def test_model_correctness(self, kernel_dir: str = None) -> Dict[str, Dict[str, bool]]:
        """Test full model correctness for all models and configurations.

        This method runs each model with each test configuration, comparing
        eager mode vs backend mode to verify numerical correctness.

        Args:
            kernel_dir: Optional directory containing kernels for backend

        Returns:
            Dictionary mapping model_name -> {config_name -> bool}
            where bool indicates if the test passed
        """
        results = {}

        for model in self.models:
            model_results = {}

            # Test each configuration for this model
            if "model_tests" not in model["config"]:
                logger.warning(f"Model {model['name']} has no model_tests in config")
                continue

            # model_tests is a dict mapping test_name -> serialized_args
            for test_name, test_args in model["config"]["model_tests"].items():
                test = FullModelTest(
                    model_name=model["name"],
                    model_class=model["class"],
                    model_config=model["config"].get("model_config", {}),
                    test_name=test_name,
                    test_args=test_args,
                )

                is_correct = test.test_correctness(kernel_dir=kernel_dir)
                model_results[test_name] = is_correct

                status = "PASS" if is_correct else "FAIL"
                logger.info(f"{model['name']}::{test_name}: {status}")

            results[model["name"]] = model_results

        return results

    def print_model_correctness_results(self, results: Dict[str, Dict[str, bool]]):
        """Print formatted model correctness results.

        Args:
            results: Dictionary mapping model_name -> {test_name -> bool}
        """
        print("\n" + "=" * 80)
        print("FULL MODEL TESTING")
        print("=" * 80)
        print("\nModel Correctness Results:")
        print("-" * 80)

        total_passed = 0
        total_tests = 0

        for model_name, test_results in results.items():
            passed = sum(1 for result in test_results.values() if result)
            total = len(test_results)
            total_passed += passed
            total_tests += total
            percentage = (passed / total * 100) if total > 0 else 0
            print(f"  {model_name}: {passed}/{total} configs passed ({percentage:.1f}%)")

            # Show individual config results
            for config_name, is_correct in test_results.items():
                status = "✓ PASS" if is_correct else "✗ FAIL"
                print(f"    {config_name}: {status}")

        print("-" * 80)
        if total_tests > 0:
            overall_percentage = total_passed / total_tests * 100
            print(f"\nModel Suite Score: {total_passed}/{total_tests} ({overall_percentage:.1f}%)")
        else:
            print("\nNo model tests were run")
        print("=" * 80)
