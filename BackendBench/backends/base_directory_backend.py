# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import importlib.util
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional

from torch._library.custom_ops import CustomOpDef

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
        self.ops_dir = Path(ops_dir)
        self.compiled_kernels: dict[str, Callable | CustomOpDef] = {}
        self._load_kernels()

    def _load_kernels(self):
        """Load all kernel implementations from the operations directory."""
        if not os.path.exists(self.ops_dir):
            logger.warning(f"ops directory {self.ops_dir} does not exist")
            return

        loaded_count = 0
        for op_dir in self.ops_dir.iterdir():
            if not op_dir.is_dir():
                continue
            op_name = op_dir.name
            try:
                loaded_count += len(self.load_op_implementations(op_name, op_dir))
            except Exception as e:
                logger.error(f"Error loading {op_name}: {e}")

        logger.info(f"{self.name} loaded {loaded_count} implementations from {self.ops_dir}/")

    @abstractmethod
    def load_op_implementations(self, op_name: str, op_dir: Path) -> list[str]:
        """
        Load all kernel implementations from the operations directory.
        """
        pass

    @classmethod
    def load_py_kernel_from_file(cls, file_path: Path, op_name: str) -> Callable:
        """
        Dynamically load a kernel implementation function from a Python file.

        Each python file should contain implementation files that export a function
        named {op_name}_kernel_impl.

        Args:
            file_path: Path to the Python implementation file
            op_name: Base name of the operator (e.g., "add", "mul", "conv2d")

        Returns:
            Callable kernel implementation function

        Raises:
            ValueError: If the expected kernel function is not found in the file
        """
        return cls._load_py_symbol_from_file(file_path, f"{op_name}_kernel_impl")
    
    @classmethod
    @lru_cache(maxsize=1)  # when we load different symbols from same file this could be called multiple times
    def _load_py_file(cls, file_path: Path) -> Callable:
        # todo: error handling?
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    @classmethod 
    def _load_py_symbol_from_file(cls, file_path: Path, symbol_name: str) -> Callable:
        module = cls._load_py_file(file_path)
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name)
        else:
            raise ValueError(f"No symbol named {symbol_name} found in {file_path}")

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        raise KeyError(
            f"Operator {key} not implemented in {self.name} - add implementation to {self.ops_dir}/"
        )

    def __contains__(self, key):
        return key in self.compiled_kernels
