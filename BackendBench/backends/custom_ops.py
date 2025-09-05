# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Callable, List

from .directory_backend_abs import BaseDirectoryBackendABS

logger = logging.getLogger(__name__)


class CustomOpsBackend(BaseDirectoryBackendABS):
    """
    Directory-based backend for non-ATen custom operators.
    
    Discovers and loads kernel implementations from the custom ops directory structure.
    Each operator directory should contain Python files with kernel implementations
    following the naming pattern: {op_name}_*.py (excluding gen_input.py and reference files).
    
    The backend registers each implementation as op__impl_name for testing.
    """

    def __init__(self, ops_dir: str = "custom_ops"):
        super().__init__("custom_ops", ops_dir)

    def _discover_implementation_files(self, op_name: str, op_dir: str) -> List[str]:
        """
        Discover implementation files for a custom operator.
        
        Looks for all .py files except gen_input.py and reference files.
        """
        impl_files = [
            f
            for f in os.listdir(op_dir)
            if f.endswith(".py") 
            and f != "gen_input.py" 
            and not f.startswith(f"{op_name}_reference")
        ]
        return impl_files

    def _register_implementation(self, op_name: str, impl_file: str, kernel_func: Callable) -> List[str]:
        """
        Register a kernel implementation for a custom operator.
        
        Registers the implementation as op__impl_name where impl_name is the filename
        without the .py extension.
        """
        impl_name = impl_file[:-3]  # Remove .py extension
        key = f"{op_name}__{impl_name}"
        self.compiled_kernels[key] = kernel_func
        return [key]