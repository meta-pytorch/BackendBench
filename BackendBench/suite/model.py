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

import json
import os
import importlib.util
import logging
import torch
from typing import Dict, List, Any, Optional

from BackendBench.utils import get_pytorch_op
from .torchbench import TorchBenchTestSuite, TorchBenchTest

logger = logging.getLogger(__name__)


def load_toy_models(toy_models_dir: str = "models", filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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
        logger.warning(f"Toy models directory not found: {toy_models_dir}")
        return models

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
            logger.warning(f"Model file not found: {model_file}")
            continue

        if not os.path.exists(config_file):
            logger.warning(f"Config file not found: {config_file}")
            continue

        try:
            # Load config
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Load model class dynamically
            spec = importlib.util.spec_from_file_location(model_name, model_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find model class (ends with "Model")
            model_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    attr_name.endswith("Model") and
                    hasattr(attr, "forward")):
                    model_class = attr
                    break

            if model_class is None:
                logger.error(f"No model class found in {model_file}")
                continue

            models.append({
                "name": model_name,
                "class": model_class,
                "config": config
            })
            logger.info(f"Loaded model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue

    return models


def _create_torchbench_op_test(model_name: str, model_class, config: Dict[str, Any], op_name: str):
    """Create a TorchBenchOpTest for a specific operator from a toy model.

    Args:
        model_name: Name of the model
        model_class: Model class
        config: Configuration dictionary
        op_name: Operator name (e.g., "conv2d", "relu")

    Returns:
        TorchBenchOpTest instance compatible with existing evaluation infrastructure
    """
    from .torchbench import TorchBenchOpTest
    from BackendBench.utils import serialize_args

    # Generate test inputs from model configs
    inputs = []
    for test_config in config["test_configs"]:
        # Extract input shape from test config
        forward_args = test_config["forward_args"]
        batch_size = forward_args["batch_size"]
        input_shape = forward_args["input_shape"]

        # Create input tensor
        full_shape = [batch_size] + input_shape
        input_tensor = torch.randn(*full_shape)

        # Serialize the input for TorchBenchOpTest
        # TorchBenchOpTest expects serialized inputs (strings)
        serialized = serialize_args([input_tensor], {})
        inputs.append(serialized)

    # Create TorchBenchOpTest - it expects op name and inputs
    # We need to convert op_name to full torch.ops format
    op = get_pytorch_op(op_name)
    if op is None:
        raise ValueError(f"Could not find PyTorch operation for {op_name}")

    # Get the full op string (e.g., "aten.conv2d.default")
    op_str = str(op).replace("torch.ops.", "")

    # Create the test with serialized inputs
    return TorchBenchOpTest(op_str, inputs, topn=None)


class FullModelTest:
    """Complete model forward/backward testing.

    This class handles running a model with a specific test configuration
    in both eager mode and backend mode, then comparing the results.
    """

    def __init__(self, model_name: str, model_class, config: Dict[str, Any], test_config: Dict[str, Any]):
        """Initialize FullModelTest.

        Args:
            model_name: Name of the model being tested
            model_class: Model class to instantiate
            config: Full model configuration including model_config
            test_config: Specific test configuration with forward_args
        """
        self.model_name = model_name
        self.model_class = model_class
        self.config = config
        self.test_config = test_config

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

        # Extract model configuration
        model_config = self.config["model_config"]["init_args"]

        # Extract input configuration
        forward_args = self.test_config["forward_args"]
        batch_size = forward_args["batch_size"]
        input_shape = forward_args["input_shape"]

        # Create full input shape: [batch_size, *input_shape]
        full_shape = [batch_size] + input_shape

        # Set seed for deterministic behavior
        seed = model_config.get("seed", 42)
        torch.manual_seed(seed)

        # Create fresh model instance
        model = self.model_class(**model_config)
        model.train()

        # Create input tensor with requires_grad for input gradient
        x = torch.randn(*full_shape, requires_grad=True)

        # Run forward + backward with or without backend
        if backend_enabled:
            # Use context manager to enable backend
            if kernel_dir is None:
                # Default to generated_kernels directory
                kernel_dir = os.path.join(os.getcwd(), "generated_kernels")

            with BackendBench.BackendBench.enable(kernel_dir=kernel_dir):
                output = model(x)
                loss = output.sum()
                loss.backward()
        else:
            # Run in eager mode (no backend)
            output = model(x)
            loss = output.sum()
            loss.backward()

        # Collect gradients: [input_grad, param1_grad, param2_grad, ...]
        grads = []

        # Input gradient
        if x.grad is not None:
            grads.append(x.grad.clone())

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
                logger.debug(f"{self.model_name}::{self.test_config['name']}: Output mismatch")
                return False

            # Compare number of gradients
            if len(eager_grads) != len(backend_grads):
                logger.debug(
                    f"{self.model_name}::{self.test_config['name']}: "
                    f"Gradient count mismatch ({len(eager_grads)} vs {len(backend_grads)})"
                )
                return False

            # Compare each gradient
            for i, (eager_grad, backend_grad) in enumerate(zip(eager_grads, backend_grads)):
                if not torch.allclose(eager_grad, backend_grad, atol=atol, rtol=rtol):
                    logger.debug(
                        f"{self.model_name}::{self.test_config['name']}: "
                        f"Gradient {i} mismatch"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"{self.model_name}::{self.test_config['name']}: Correctness test failed: {e}")
            return False


class ModelSuite(TorchBenchTestSuite):
    """Model Suite extending TorchBenchTestSuite.

    Provides two testing approaches:
    1. Operator-level testing via __iter__() (inherited infrastructure)
    2. Model-level correctness testing via test_model_correctness() (Model Suite specific)
    """

    def __init__(self, name: str = "model", filter: Optional[List[str]] = None, models_dir: str = None):
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

        This method enables operator-level testing using the inherited
        TorchBench infrastructure. Returns TorchBenchOpTest instances.
        """
        for model in self.models:
            # Extract operators from config
            if "expected_operators" not in model["config"]:
                logger.warning(f"Model {model['name']} has no expected_operators in config")
                continue

            expected_ops = model["config"]["expected_operators"]

            # Yield forward pass operators
            if "forward_pass" in expected_ops:
                for op_name in expected_ops["forward_pass"]:
                    try:
                        yield _create_torchbench_op_test(model["name"], model["class"], model["config"], op_name)
                    except Exception as e:
                        logger.error(f"Failed to create test for forward op {op_name}: {e}")

            # Yield backward pass operators
            if "backward_pass" in expected_ops:
                for op_name in expected_ops["backward_pass"]:
                    try:
                        yield _create_torchbench_op_test(model["name"], model["class"], model["config"], op_name)
                    except Exception as e:
                        logger.error(f"Failed to create test for backward op {op_name}: {e}")

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
            for test_config in model["config"]["test_configs"]:
                test = FullModelTest(
                    model_name=model["name"],
                    model_class=model["class"],
                    config=model["config"],
                    test_config=test_config
                )

                test_name = test_config["name"]
                is_correct = test.test_correctness(kernel_dir=kernel_dir)
                model_results[test_name] = is_correct

                status = "PASS" if is_correct else "FAIL"
                logger.info(f"{model['name']}::{test_name}: {status}")

            results[model["name"]] = model_results

        return results
