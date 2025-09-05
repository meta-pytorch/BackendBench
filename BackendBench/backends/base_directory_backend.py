# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from .base import Backend

logger = logging.getLogger(__name__)


class BaseDirectoryBackendABS(Backend, ABC):
    """
    Abstract base class for directory-based backends.
    
    Provides common functionality for discovering and loading kernel implementations
    from filesystem directories. Subclasses can customize the discovery and loading
    behavior by implementing abstract methods.
    
    This is an abstract class that cannot be instantiated directly. Use concrete
    implementations like DirectoryBackend or CustomOpsBackend instead.
    """

    def __init__(self, name: str, ops_dir: str):
        super().__init__(name)
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self._load_kernels()

    def _load_kernels(self):
        """Load all kernel implementations from the operations directory."""
        if not os.path.exists(self.ops_dir):
            logger.warning(f"ops directory {self.ops_dir} does not exist")
            return

        loaded_count = 0
        for op_name in os.listdir(self.ops_dir):
            op_dir = os.path.join(self.ops_dir, op_name)
            if not os.path.isdir(op_dir):
                continue

            try:
                loaded_count += self._load_op_implementations(op_name, op_dir)
            except Exception as e:
                logger.error(f"Error loading {op_name}: {e}")

        logger.info(f"{self.name} loaded {loaded_count} implementations from {self.ops_dir}/")

    def _load_op_implementations(self, op_name: str, op_dir: str) -> int:
        """
        Load all implementations for a specific operator.
        
        Args:
            op_name: Name of the operator
            op_dir: Path to the operator directory
            
        Returns:
            Number of implementations loaded
        """
        impl_files = self._discover_implementation_files(op_name, op_dir)
        if not impl_files:
            logger.debug(f"No implementation files found in {op_dir}")
            return 0

        loaded_count = 0
        for impl_file in impl_files:
            impl_path = os.path.join(op_dir, impl_file)
            
            try:
                kernel_func = self._load_kernel_from_file(impl_path, op_name)
                keys = self._register_implementation(op_name, impl_file, kernel_func)
                for key in keys:
                    logger.info(f"Loaded {op_name} from {impl_file} -> {key}")
                loaded_count += len(keys)
            except Exception as e:
                logger.error(f"Error loading {op_name} from {impl_file}: {e}")

        return loaded_count

    @abstractmethod
    def _discover_implementation_files(self, op_name: str, op_dir: str) -> List[str]:
        """
        Discover implementation files for an operator.
        
        Args:
            op_name: Name of the operator
            op_dir: Path to the operator directory
            
        Returns:
            List of implementation file names
        """
        pass

    @abstractmethod
    def _register_implementation(self, op_name: str, impl_file: str, kernel_func: Callable) -> List[str]:
        """
        Register a kernel implementation and return the keys it was registered under.
        
        Args:
            op_name: Name of the operator
            impl_file: Name of the implementation file
            kernel_func: The kernel function to register
            
        Returns:
            List of keys the implementation was registered under
        """
        pass

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
        raise KeyError(
            f"Operator {key} not implemented in {self.name} - add implementation to {self.ops_dir}/"
        )

    def __contains__(self, key):
        return key in self.compiled_kernels
