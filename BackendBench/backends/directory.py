# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Callable, List

from .base_directory_backend import BaseDirectoryBackendABS
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

    def load_op_implementations(self, op_name: str, op_dir: Path) -> list[str]:
        """
        Load all kernel implementations from the operations directory.
        """
        impl_files = [
            f for f in op_dir.iterdir()
            if f.is_file()
            and f.suffix == ".py"
        ]

        op_variants = query(op_name)
        registered_keys = []

        kernel_func = self.load_py_kernel_from_file(sorted(impl_files)[0], op_name)

        if op_variants:
            for variant_info in op_variants:
                op_full_name = variant_info["op"]
                pytorch_op = get_pytorch_op(op_full_name)
                if pytorch_op:
                    self.compiled_kernels[pytorch_op] = kernel_func
                    registered_keys.append(str(pytorch_op))
        
        if not registered_keys:
            logger.warning(f"Could not find operator variants for {op_name} in op_map")

        return registered_keys
