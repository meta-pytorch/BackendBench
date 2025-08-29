# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import os
from typing import Callable, Dict


from .base import Backend
from ..scripts.op_map import query
import torch

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

        This method scans the ops_dir for subdirectories named after PyTorch operators
        (e.g., "add", "mul", "conv2d"). Each subdirectory should contain Python files
        with kernel implementations following the naming pattern: {op_name}_implementation*.py

        The loading process:
        1. Finds implementation files in each operator directory
        2. Uses the authoritative op_map.query() to discover all PyTorch operator variants
           that should map to this directory (e.g., add.Tensor, add_.Scalar, add.out)
        3. Registers the same kernel implementation for all discovered variants
        4. This ensures comprehensive coverage - a single "add" implementation handles
           all add variants: functional (add.Tensor), in-place (add_.Tensor), out (add.out)
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
                if f.endswith(".py") and f.startswith(f"{op_name}_implementation")
            ]
            if not impl_files:
                logger.debug(f"No implementation files found in {op_dir}")
                continue

            impl_file = sorted(impl_files)[0]
            impl_path = os.path.join(op_dir, impl_file)

            try:
                kernel_func = self._load_kernel_from_file(impl_path, op_name)
                op_variants = query(op_name)

                if op_variants:
                    for variant_info in op_variants:
                        op_full_name = variant_info["op"]
                        pytorch_op = self._get_pytorch_op(op_full_name)
                        if pytorch_op:
                            self.compiled_kernels[pytorch_op] = kernel_func
                            logger.info(f"Loaded {op_name} from {impl_file} -> {op_full_name}")

                    loaded_count += 1
                else:
                    logger.warning(f"Could not find operator variants for {op_name} in op_map")

            except Exception as e:
                logger.error(f"Error loading {op_name} from {impl_file}: {e}")

        logger.info(f"DirectoryBackend loaded {loaded_count} kernels from {self.ops_dir}/")

    def _get_pytorch_op(self, op_name: str):
        """
        Convert an operator name string to the actual PyTorch operation object.

        PyTorch operations are structured as torch.ops.aten.{base_name}.{overload}.
        This method handles the conversion from string names like "add.Tensor" or "relu.default"
        to the actual callable operation objects that can be registered in the dispatcher.

        Args:
            op_name: String name like "add.Tensor", "relu.default", or just "relu"

        Returns:
            PyTorch operation object that can be used for dispatch registration,
            or None if the operation doesn't exist in torch.ops.aten
        """
        try:
            if "." in op_name:
                base_name, overload = op_name.split(".", 1)
                if overload == "default":
                    return getattr(torch.ops.aten, base_name).default
                else:
                    return getattr(getattr(torch.ops.aten, base_name), overload)
            else:
                return getattr(torch.ops.aten, op_name).default
        except AttributeError:
            logger.warning(f"Could not find PyTorch operation for {op_name}")
            return None

    def _load_kernel_from_file(self, file_path: str, op_name: str) -> Callable:
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
        # Fallback to original operation if not implemented
        return key

    def __contains__(self, key):
        return key in self.compiled_kernels
