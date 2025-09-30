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
        model_dir = os.path.join(toy_models_dir, model_name)
        if not os.isdir(model_dir):
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

        # Extract operators from model configs
        model_ops = set()
        for model in models:
            config_ops = model["config"].get("ops", [])
            if not config_ops:
                raise ValueError(f"Model {model['name']} has no 'ops' field in config")
            model_ops.update(config_ops)
            logger.info(f"Model {model['name']}: {len(config_ops)} operators defined in config")

        logger.info(f"ModelSuite: Total {len(model_ops)} unique operators across all models")

        # Get torchbench ops and filter
        torchbench_ops = _get_torchbench_ops()

        # Filter torchbench ops to only include those in model configs
        filtered_ops = {}
        unsupported_ops = []
        for model_op in model_ops:
            # Find matching torchbench ops
            matched = False
            for op_name, op_inputs in torchbench_ops.items():
                if model_op in op_name:
                    filtered_ops[op_name] = op_inputs
                    matched = True
            if not matched:
                unsupported_ops.append(model_op)

        # Error out if any ops are not supported by torchbench
        if unsupported_ops:
            raise ValueError(
                f"The following operators are not supported by TorchBench: {unsupported_ops}"
            )

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
