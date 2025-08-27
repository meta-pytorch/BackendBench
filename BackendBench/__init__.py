# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
BackendBench: A PyTorch backend evaluation framework.
"""

import torch
import os
import warnings
from typing import Optional, Union
from pathlib import Path

# Import the existing DirectoryBackend implementation
from BackendBench.backends.directory import DirectoryBackend


__version__ = "0.1.0"
__all__ = ["enable", "disable"]

# Global state
_lib = None


def _monkey_patch_operators(op_custom_impl):
    """
    Replace PyTorch operators with custom implementations using torch.library.
    """
    global _lib

    # Use torch.library to register custom implementations
    _lib = torch.library.Library("aten", "IMPL", "CPU")

    patched_count = 0
    for op, custom_impl in op_custom_impl.items():
        try:
            # Extract operator name and overload from the OpOverload
            op_name = op._schema.name
            overload_name = op._schema.overload_name

            # Create the full name for torch.library
            if overload_name:
                full_name = f"{op_name}.{overload_name}"
            else:
                full_name = op_name

            # Register the custom implementation
            _lib.impl(full_name, custom_impl, "CPU")
            patched_count += 1

        except Exception as e:
            # Some operators might not be patchable
            warnings.warn(f"Could not register {op}: {e}")

    if patched_count > 0:
        print(f"Successfully registered {patched_count} custom operators")


def enable(
    kernel_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> None:
    """
    Enable the DirectoryBackend to use custom operator implementations.
    """
    global _lib

    if _lib is not None:
        warnings.warn("DirectoryBackend is already enabled. Call disable() first.")
        return

    # Set default kernel directory
    if kernel_dir is None:
        kernel_dir = os.path.join(os.getcwd(), "generated_kernels")

    kernel_dir = Path(kernel_dir)

    # Check if kernel directory exists
    if not kernel_dir.exists():
        warnings.warn(
            f"Kernel directory {kernel_dir} does not exist. Call"
            f"directory_backend.setup_operators('{kernel_dir}') manually."
        )
        return

    # Initialize the backend
    try:
        _current_backend = DirectoryBackend(str(kernel_dir))

        # Actually monkey-patch PyTorch operators
        _monkey_patch_operators(_current_backend.compiled_kernels)

        if verbose:
            print(f"DirectoryBackend enabled with kernel directory: {kernel_dir}")
            print(f"Loaded {len(_current_backend.compiled_kernels)} custom operators")

    except Exception as e:
        warnings.warn(f"Failed to enable DirectoryBackend: {e}")
        disable()


def disable() -> None:
    """
    Disable the DirectoryBackend and restore original PyTorch operators.
    """
    global _lib

    if _lib is None:
        warnings.warn("DirectoryBackend is not currently enabled")
        return

    # Restore original operators
    _lib = None
    print("DirectoryBackend disabled")
