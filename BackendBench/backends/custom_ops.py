# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import os
from typing import Callable, Dict

import torch

from .base import Backend

logger = logging.getLogger(__name__)


class CustomOpsBackend(Backend):
    """
    Filesystem-based custom backend for non-ATen operators.

    Layout:
      ./custom_ops/<op>/<impl_name>.py

    Each implementation file should export a function named {op}_kernel_impl.
    The backend discovers all .py files in each op directory and registers them
    as separate implementations (op__impl_name).
    """

    def __init__(self, ops_dir: str = "custom_ops"):
        super().__init__("custom_ops")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self._load_kernels()

    def _load_kernels(self):
        """
        Discovers and loads kernel implementations from the custom ops directory structure.

        This method scans the ops_dir for subdirectories named after custom operators.
        Each subdirectory should contain Python files with kernel implementations
        following the naming pattern: {op_name}_kernel_impl.

        The loading process:
        1. Finds all .py files in each operator directory (excluding gen_input.py)
        2. Loads each file and looks for {op_name}_kernel_impl function
        3. Registers implementations as op__impl_name for testing
        """
        if not os.path.exists(self.ops_dir):
            logger.warning(f"ops directory {self.ops_dir} does not exist")
            return

        loaded_count = 0
        for op_name in os.listdir(self.ops_dir):
            op_dir = os.path.join(self.ops_dir, op_name)
            if not os.path.isdir(op_dir):
                continue

            impl_files = [
                f
                for f in os.listdir(op_dir)
                if f.endswith(".py") and f != "gen_input.py" and not f.startswith(f"{op_name}_reference")
            ]
            if not impl_files:
                logger.debug(f"No implementation files found in {op_dir}")
                continue

            for impl_file in impl_files:
                impl_path = os.path.join(op_dir, impl_file)
                impl_name = impl_file[:-3]  # Remove .py extension

                try:
                    kernel_func = self._load_kernel_from_file(impl_path, op_name)
                    key = f"{op_name}__{impl_name}"
                    self.compiled_kernels[key] = kernel_func
                    logger.info(f"Loaded {op_name} implementation from {impl_file} -> {key}")
                    loaded_count += 1

                except Exception as e:
                    logger.error(f"Error loading {op_name} from {impl_file}: {e}")

        logger.info(f"CustomOpsBackend loaded {loaded_count} implementations from {self.ops_dir}/")

    def _load_kernel_from_file(self, file_path: str, op_name: str) -> Callable:
        """
        Dynamically load a kernel implementation function from a Python file.

        Each implementation file should export a function named {op_name}_kernel_impl.

        Args:
            file_path: Path to the Python implementation file
            op_name: Base name of the operator (e.g., "myop")

        Returns:
            Callable kernel implementation function

        Raises:
            ValueError: If the expected kernel function is not found in the file
        """
        spec = importlib.util.spec_from_file_location(f"op_{op_name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel_func_name = f"{op_name}_kernel_impl"
        if hasattr(module, kernel_func_name):
            return getattr(module, kernel_func_name)
        else:
            raise ValueError(f"No function named {kernel_func_name} found in {file_path}")

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        raise KeyError(
            f"Operator {key} not implemented in CustomOpsBackend - add implementation to {self.ops_dir}/"
        )

    def __contains__(self, key):
        return key in self.compiled_kernels