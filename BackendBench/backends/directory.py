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


class DirectoryBackend(Backend):
    def __init__(self, ops_dir="generated_kernels"):
        super().__init__("directory")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self._load_kernels()

    def _load_kernels(self):
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

            # Use the first implementation file
            impl_file = sorted(impl_files)[0]  # Sort to ensure consistent selection
            impl_path = os.path.join(op_dir, impl_file)

            try:
                # Load the implementation and map to PyTorch operation
                kernel_func = self._load_kernel_from_file(impl_path, op_name)
                pytorch_ops = self._find_pytorch_ops(op_name)

                if pytorch_ops:
                    for pytorch_op in pytorch_ops:
                        self.compiled_kernels[pytorch_op] = kernel_func
                        logger.info(f"Loaded {op_name} from {impl_file} -> {pytorch_op}")
                    loaded_count += 1
                else:
                    logger.warning(f"Could not map {op_name} to PyTorch operation")

            except Exception as e:
                logger.error(f"Error loading {op_name} from {impl_file}: {e}")

        logger.info(f"DirectoryBackend loaded {loaded_count} kernels from {self.ops_dir}/")

    def _load_kernel_from_file(self, file_path: str, op_name: str) -> Callable:
        spec = importlib.util.spec_from_file_location(f"op_{op_name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel_func_name = f"{op_name}_kernel_impl"
        if hasattr(module, kernel_func_name):
            return getattr(module, kernel_func_name)
        else:
            raise ValueError(f"No function named {kernel_func_name} found in {file_path}")

    def _find_pytorch_ops(self, op_name: str):
        """Map operation name to PyTorch operations.

        Returns a list of PyTorch operations that match the directory name.
        This handles the common case where a directory name like 'add' should map
        to multiple overloads like add.default, add.Tensor, etc.
        """
        matched_ops = []

        # Handle suffixed directory names (e.g., add_out -> add.out)
        base_name = op_name
        suffix = None
        if "_" in op_name:
            parts = op_name.rsplit("_", 1)
            if parts[1] in ["out", "inplace", "scalar"]:
                base_name = parts[0]
                suffix = parts[1]

        # Try to find the operation in torch.ops.aten
        if hasattr(torch.ops.aten, base_name):
            aten_op = getattr(torch.ops.aten, base_name)

            # If we have a specific suffix, try to get that overload
            if suffix and hasattr(aten_op, suffix):
                matched_ops.append(getattr(aten_op, suffix))
            else:
                # Otherwise, try common overloads
                for overload in ["default", "Tensor", "Scalar", "int", "float"]:
                    if hasattr(aten_op, overload):
                        op = getattr(aten_op, overload)
                        matched_ops.append(op)

        # Also check for operations that might be in other namespaces
        # This could be extended based on actual usage patterns

        return matched_ops

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        # Fallback to original operation if not implemented
        return key

    def __contains__(self, key):
        return key in self.compiled_kernels
