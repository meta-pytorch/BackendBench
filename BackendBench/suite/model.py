# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Model Suite for testing models defined in configs.
"""

import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Optional

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


class ModelSuite:
    """Model Suite for end-to-end model testing."""

    def __init__(
        self,
        name: str = "model",
        filter: Optional[List[str]] = None,
    ):
        """Initialize ModelSuite.

        Args:
            name: Suite name (default: "model")
            filter: Optional list of model names to load
        """
        models_dir = os.path.join(os.path.dirname(__file__), "models")

        # Load models
        models = load_models(models_dir=models_dir, filter=filter)
        logger.info(f"ModelSuite: Loaded {len(models)} models from {models_dir}")

        # Store loaded models
        self.models = models
        self.name = name
