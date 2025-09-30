# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Model Suite for testing operators traced from toy models.

This suite extends TorchBenchTestSuite by tracing model execution
to extract operators, then filtering the TorchBench dataset to only
include those operators.
"""

import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set

import torch

from BackendBench.data_loaders import load_ops_from_source, op_list_to_benchmark_dict

from .torchbench import TorchBenchTestSuite

logger = logging.getLogger(__name__)

# Cache for torchbench ops to avoid reloading
_TORCHBENCH_OPS_CACHE = None


def _get_torchbench_ops():
    """Get list of available ops from torchbench dataset (cached)."""
    global _TORCHBENCH_OPS_CACHE
    if _TORCHBENCH_OPS_CACHE is None:
        ops_list = load_ops_from_source(source=None, format="parquet")
        _TORCHBENCH_OPS_CACHE = op_list_to_benchmark_dict(ops_list)
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
        if not os.isdir(model_dir):
            continue

        # Strict naming convention: folder_name/folder_name.py and folder_name/folder_name.json
        model_file = os.path.join(model_dir, f"{model_name}.py")
        config_file = os.path.join(model_dir, f"{model_name}.json")

        # Check both files exist
        if not os.path.exists(model_file):
            if filter is not None and model_name in filter:
                raise FileNotFoundError(f"Model file not found: {model_file}")
            logger.warning(f"Model file not found: {model_file}")
            continue

        if not os.path.exists(config_file):
            if filter is not None and model_name in filter:
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

            models.append({"name": model_name, "class": model_class, "config": config})
            logger.info(f"Loaded model: {model_name}")

        except Exception as e:
            if filter is not None and model_name in filter:
                raise RuntimeError(f"Failed to load model {model_name}: {e}")
            logger.error(f"Failed to load {model_name}: {e}")
            continue

    # If a filter was specified but no models were loaded, raise an error
    if filter is not None and len(models) == 0:
        raise ValueError(f"No models found matching filter: {filter}")

    return models


def _trace_model_ops(model_class, model_config: Dict[str, Any]) -> Set[str]:
    """Trace model execution to extract operator names.

    Args:
        model_class: Model class to instantiate
        model_config: Model configuration dict with init_args and model_tests

    Returns:
        Set of operator names in aten format (e.g., "aten.conv2d.default")
    """
    import torch._dynamo as dynamo

    from BackendBench.utils import deserialize_args

    init_args = model_config.get("init_args", {})
    model = model_class(**init_args)
    model.eval()

    # Get first test input to trace with
    model_tests = model_config.get("model_tests", {})
    if not model_tests:
        raise ValueError("No model_tests found in config")

    first_test = next(iter(model_tests.values()))
    args, kwargs = deserialize_args(first_test)

    # Trace the model to extract ops
    ops = set()

    def capture_ops(gm, example_inputs):
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target = node.target
                if hasattr(target, "__module__") and "torch.ops" in target.__module__:
                    ops.add(str(target))
        return gm

    with torch.no_grad():
        try:
            compiled_model = dynamo.optimize(capture_ops)(model)
            compiled_model(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to trace model: {e}")

    return ops


class ModelSuite(TorchBenchTestSuite):
    """Model Suite that filters TorchBench operators based on model tracing.

    This suite traces model execution to extract operators, then creates
    a filtered TorchBench suite containing only those operators.
    """

    def __init__(
        self,
        name: str = "model",
        filter: Optional[List[str]] = None,
        models_dir: str = None,
        topn: Optional[int] = None,
    ):
        """Initialize ModelSuite.

        Args:
            name: Suite name (default: "model")
            filter: Optional list of model names to load
            models_dir: Optional directory for models (default: "BackendBench/suite/models")
            topn: Optional limit on number of tests per operator
        """
        # Default to models under suite/models
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")

        # Load models
        models = load_toy_models(toy_models_dir=models_dir, filter=filter)
        logger.info(f"ModelSuite: Loaded {len(models)} models from {models_dir}")

        # Trace models to extract operators
        model_ops = set()
        for model in models:
            try:
                ops = _trace_model_ops(model["class"], model["config"])
                model_ops.update(ops)
                logger.info(f"Model {model['name']}: Found {len(ops)} operators")
            except Exception as e:
                logger.warning(f"Failed to trace model {model['name']}: {e}")

        logger.info(f"ModelSuite: Total {len(model_ops)} unique operators across all models")

        # Get torchbench ops and filter
        torchbench_ops = _get_torchbench_ops()

        # Convert model ops to the format used in torchbench (strip <built-in function ...>)
        # Example: "<built-in function conv2d>" -> "aten.conv2d.default"
        filtered_ops = {}
        for op_name, op_inputs in torchbench_ops.items():
            # Check if any model op matches this torchbench op
            # Model ops from dynamo are like "aten.conv2d.default"
            if any(model_op in op_name for model_op in model_ops):
                filtered_ops[op_name] = op_inputs

        if not filtered_ops:
            raise ValueError(
                f"No operators from models found in TorchBench dataset. "
                f"Model operators: {model_ops}"
            )

        logger.info(
            f"ModelSuite: Filtered to {len(filtered_ops)} operators "
            f"(from {len(torchbench_ops)} total)"
        )

        # Initialize parent class with filtered ops
        self.name = name
        self.topn = topn
        self.optests = filtered_ops

        # Deduplicate strings in self.optests
        for op in self.optests:
            self.optests[op] = list(set(self.optests[op]))
