# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
BackendBench: A PyTorch backend evaluation framework.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch

# Import the existing DirectoryBackend implementation
from BackendBench.backends.directory import DirectoryBackend

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = ["enable", "disable", "BackendBench"]

# Global state
_lib = None


class _BackendBenchContext:
    """Context manager for BackendBench that enables on entry and disables on exit."""
    
    def __init__(self, kernel_dir=None, namespace="aten", dispatch_key="CUDA"):
        self.kernel_dir = kernel_dir
        self.namespace = namespace
        self.dispatch_key = dispatch_key
        self._was_enabled = _lib is not None
    
    def __enter__(self):
        enable(self.kernel_dir, self.namespace, self.dispatch_key)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._was_enabled:
            disable()


class BackendBench:
    """BackendBench main class with context manager support."""
    
    @classmethod
    def enable(cls, kernel_dir=None, namespace="aten", dispatch_key="CUDA"):
        """
        Return a context manager that enables BackendBench on entry and disables on exit.
        
        Args:
            kernel_dir: Path to the directory containing custom kernels
            namespace: PyTorch namespace to patch (default: "aten")
            dispatch_key: Dispatch key for the kernels (default: "CUDA")
            
        Returns:
            Context manager that can be used with 'with' statement
            
        Example:
            with BackendBench.enable(generated_kernels):
                model.forward()  # uses LLM kernels
            # On exit, uses aten kernels
        """
        return _BackendBenchContext(kernel_dir, namespace, dispatch_key)


def _monkey_patch_operators(op_custom_impl, namespace="aten", dispatch_key="CUDA"):
    """
    Replace PyTorch operators with custom implementations using torch.library.
    """
    global _lib

    assert dispatch_key in ["CPU", "CUDA"], "Only CPU and CUDA dispatch keys are supported"

    # Use torch.library to register custom implementations
    if _lib is None:
        _lib = torch.library.Library(namespace, "IMPL", dispatch_key)

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
            _lib.impl(full_name, custom_impl, dispatch_key)
            patched_count += 1

        except Exception as e:
            # Some operators might not be patchable
            logger.warning(f"Could not register {op}: {e}")

    if patched_count > 0:
        print(f"Successfully registered {patched_count} custom operators")
    else:
        print("No custom operators registered")


def enable(
    kernel_dir: Optional[Union[str, Path]] = None,
    namespace: str = "aten",
    dispatch_key: str = "CUDA",
) -> None:
    """
    Enable the DirectoryBackend to use custom operator implementations.
    """
    print("Enabling DirectoryBackend")
    # Set default kernel directory
    if kernel_dir is None:
        kernel_dir = Path(__file__).parents[1] / "generated_kernels"

    kernel_dir = Path(kernel_dir)

    # Check if kernel directory exists
    if not kernel_dir.exists():
        logger.warning(
            f"Kernel directory {kernel_dir} does not exist. Call"
            f"directory_backend.setup_operators('{kernel_dir}') manually."
        )
        return

    # Initialize the backend
    try:
        _current_backend = DirectoryBackend(str(kernel_dir))

        # Actually monkey-patch PyTorch operators
        _monkey_patch_operators(_current_backend.compiled_kernels, namespace, dispatch_key)
    except Exception as e:
        logger.warn(f"Failed to enable DirectoryBackend: {e}")
        disable()


def disable() -> None:
    """
    Disable the DirectoryBackend and restore original PyTorch operators.
    """
    global _lib

    if _lib is None:
        logger.warn("DirectoryBackend is not currently enabled")
        return

    # Restore original operators
    _lib = None
    print("DirectoryBackend disabled")
