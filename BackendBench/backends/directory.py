# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import os
from typing import Callable, Dict

from ..utils import folder_name_to_op_name, get_pytorch_op
from .base import Backend

logger = logging.getLogger(__name__)


class DirectoryBackend(Backend):
    def __init__(self, ops_dir="generated_kernels"):
        super().__init__("directory")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self._load_kernels()

    def _load_kernels(self):
        """
        Discovers and loads kernel implementations from the operator directory structure.

        This method scans the ops_dir for subdirectories named after PyTorch operator
        overloads (e.g., "add__Tensor" for add.Tensor and "add__Scalar" for add.Scalar).
        Each subdirectory should contain Python files with kernel implementations
        following the naming pattern: {op_name}_implementation*.py

        This method uses the op overload format (e.g., "add__Tensor" for "add.Tensor") and
        registers the kernel for ONLY that specific overload.
        """
        if not os.path.exists(self.ops_dir):
            logger.warning(f"ops directory {self.ops_dir} does not exist")
            return

        loaded_count = 0
        for folder_name in os.listdir(self.ops_dir):
            op_dir = os.path.join(self.ops_dir, folder_name)
            if not os.path.isdir(op_dir):
                continue

            impl_files = [
                f
                for f in os.listdir(op_dir)
                if f.endswith(".py") and f.startswith(f"{folder_name}_implementation")
            ]
            if not impl_files:
                logger.debug(f"No implementation files found in {op_dir}")
                continue

            impl_file = sorted(impl_files)[0]
            impl_path = os.path.join(op_dir, impl_file)

            try:
                op_name = folder_name_to_op_name(folder_name)
                kernel_func = self._load_kernel_from_file(impl_path, folder_name)

                pytorch_op = get_pytorch_op(op_name)
                if pytorch_op:
                    self.compiled_kernels[pytorch_op] = kernel_func
                    logger.info(f"Loaded {op_name} from {impl_file} -> {op_name}")
                    loaded_count += 1

            except Exception as e:
                logger.error(f"Error loading {op_name} from {impl_file}: {e}")

        logger.info(f"DirectoryBackend loaded {loaded_count} kernels from {self.ops_dir}/")

    def _load_kernel_from_file(self, file_path: str, folder_name: str) -> Callable:
        """
        Dynamically load a kernel implementation function from a Python file.

        Each operator directory should contain implementation files that export a function
        named {op_name}_kernel_impl. This function becomes the kernel implementation
        that gets registered for all variants of the operator.

        Args:
            file_path: Path to the Python implementation file
            op_name: Base name of the operator (e.g., "add", "mul", "conv2d")

        Returns:
            Callable kernel implementation function

        Raises:
            ValueError: If the expected kernel function is not found in the file
        """
        spec = importlib.util.spec_from_file_location(f"op_{folder_name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel_func_name = f"{folder_name}_kernel_impl"
        if hasattr(module, kernel_func_name):
            return getattr(module, kernel_func_name)
        else:
            raise ValueError(f"No function named {kernel_func_name} found in {file_path}")

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        raise KeyError(
            f"Operator {key} not implemented in DirectoryBackend - add implementation to {self.ops_dir}/"
        )

    def __contains__(self, key):
        return key in self.compiled_kernels
