# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, List

from .directory_backend_abs import BaseDirectoryBackendABS
from ..scripts.op_map import query
from ..utils import get_pytorch_op

logger = logging.getLogger(__name__)


class DirectoryBackend(BaseDirectoryBackendABS):
    """
    Directory-based backend for PyTorch ATen operations.
    
    Discovers and loads kernel implementations from the operator directory structure.
    Each operator directory should contain Python files with kernel implementations
    following the naming pattern: {op_name}_implementation*.py
    
    The loading process:
    1. Finds implementation files in each operator directory
    2. Uses the authoritative op_map.query() to discover all PyTorch operator variants
       that should map to this directory (e.g., add.Tensor, add_.Scalar, add.out)
    3. Registers the same kernel implementation for all discovered variants
    4. This ensures comprehensive coverage - a single "add" implementation handles
       all add variants: functional (add.Tensor), in-place (add_.Tensor), out (add.out)
    """

    def __init__(self, ops_dir="generated_kernels"):
        super().__init__("directory", ops_dir)

    def _discover_implementation_files(self, op_name: str, op_dir: str) -> List[str]:
        """
        Discover implementation files for an ATen operator.
        
        Looks for files matching the pattern: {op_name}_implementation*.py
        """
        import os
        impl_files = [
            f
            for f in os.listdir(op_dir)
            if f.endswith(".py") and f.startswith(f"{op_name}_implementation")
        ]
        # Return only the first implementation file (legacy behavior)
        return [sorted(impl_files)[0]] if impl_files else []

    def _register_implementation(self, op_name: str, impl_file: str, kernel_func: Callable) -> List[str]:
        """
        Register a kernel implementation for all ATen operator variants.
        
        Uses op_map.query() to discover all PyTorch operator variants that should
        map to this directory and registers the same kernel implementation for all.
        """
        op_variants = query(op_name)
        registered_keys = []

        if op_variants:
            for variant_info in op_variants:
                op_full_name = variant_info["op"]
                pytorch_op = get_pytorch_op(op_full_name)
                if pytorch_op:
                    self.compiled_kernels[pytorch_op] = kernel_func
                    registered_keys.append(str(pytorch_op))
        else:
            logger.warning(f"Could not find operator variants for {op_name} in op_map")

        return registered_keys