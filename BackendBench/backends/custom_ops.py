# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .base_directory_backend import BaseDirectoryBackendABS

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ..suite.custom_ops import CustomOpsTestSuite


class CustomOpsBackend(BaseDirectoryBackendABS):
    """
    Directory-based backend for non-ATen custom operators.

    Discovers and loads kernel implementations from the custom ops directory structure.
    Each operator directory should contain Python files with kernel implementations
    following the naming pattern: {op_name}_*.py (excluding gen_input.py and reference files).

    The backend registers each implementation as op__impl_name for testing.
    """

    def __init__(self, suite: "CustomOpsTestSuite" = None, ops_dir: str = "custom_ops"):
        self.suite = suite
        super().__init__("custom_ops", ops_dir)

    def load_op_implementations(self, op_name: str, op_dir: Path) -> list[str]:
        """
        Load all kernel implementations from the operations directory.
        """

        inputs = op_dir / "gen_input.py"
        if not inputs.exists():
            return []

        impl_files = [
            f
            for f in op_dir.iterdir()
            if f.is_file() and f.suffix == ".py" and f.name != "gen_input.py"
        ]

        ref_impl = None
        for f in impl_files:
            if f.name.startswith(f"{op_name}_reference"):
                ref_impl = f
                # we won't iter more so it's safe to remove it
                impl_files.remove(f)
                break
        else:
            logger.warning(
                f"No reference implementation found for {op_name}, using first implementation as reference"
            )
            ref_impl = impl_files[0]

        # For custom backend, each implementation is consider a seperate op to be tested.
        # Create op for each implementation and register them
        for impl in impl_files:
            impl_name = impl.stem
            impl_kernel = self.load_py_kernel_from_file(impl, op_name)
            self.compiled_kernels[impl_name] = impl_kernel

            self.suite.add_test(op_name, impl_name, [], [])

        return list(self.compiled_kernels.keys())
