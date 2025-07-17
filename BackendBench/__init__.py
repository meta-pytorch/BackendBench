"""
BackendBench: A PyTorch backend evaluation framework with monkey patching support.

Import this module to automatically monkey patch PyTorch operations with custom backends.
"""

import os
import sys
import torch
from typing import Optional, Dict, Any

from .backends import AtenBackend, FlagGemsBackend


class BackendRegistry:
    """Registry for managing different PyTorch backends."""

    def __init__(self):
        self._current_backend = None
        self._original_ops = {}
        self._patched = False

    def register_backend(self, backend_name: str, backend_instance=None):
        """Register and activate a backend."""
        if backend_instance is None:
            backend_instance = self._create_backend(backend_name)

        if self._patched:
            self.unpatch()

        self._current_backend = backend_instance
        self._patch_torch_ops()

    def _create_backend(self, backend_name: str):
        """Create a backend instance."""
        backends = {"aten": AtenBackend, "flag_gems": FlagGemsBackend}

        if backend_name not in backends:
            raise ValueError(f"Unknown backend: {backend_name}. Available: {list(backends.keys())}")

        return backends[backend_name]()

    def _patch_torch_ops(self):
        """Monkey patch torch operations with current backend."""
        if self._current_backend is None:
            return

        # Get all torch ops that the backend supports
        if hasattr(self._current_backend, "ops"):
            for torch_op, backend_impl in self._current_backend.ops.items():
                if torch_op not in self._original_ops:
                    self._original_ops[torch_op] = torch_op.default
                torch_op.default = backend_impl

        self._patched = True
        print(
            f"BackendBench: Monkey patched {len(self._original_ops)} operations with {self._current_backend.name} backend"
        )

    def unpatch(self):
        """Restore original torch operations."""
        if not self._patched:
            return

        for torch_op, original_impl in self._original_ops.items():
            torch_op.default = original_impl

        self._original_ops.clear()
        self._patched = False
        print("BackendBench: Restored original PyTorch operations")

    def get_current_backend(self):
        """Get the currently active backend."""
        return self._current_backend

    def is_patched(self):
        """Check if operations are currently patched."""
        return self._patched


# Global registry instance
_registry = BackendRegistry()


def use_backend(backend_name: str, backend_instance=None):
    """
    Switch to a different backend.

    Args:
        backend_name: Name of the backend ('aten', 'flag_gems')
        backend_instance: Optional pre-configured backend instance
    """
    _registry.register_backend(backend_name, backend_instance)


def get_backend():
    """Get the currently active backend."""
    return _registry.get_current_backend()


def restore_pytorch():
    """Restore original PyTorch operations."""
    _registry.unpatch()


def is_patched():
    """Check if BackendBench is currently patching operations."""
    return _registry.is_patched()


# Auto-configuration based on environment variables
def _auto_configure():
    """Auto-configure backend based on environment variables."""
    backend_name = os.getenv("BACKENDBENCH_BACKEND", "aten")

    try:
        use_backend(backend_name)
    except Exception as e:
        print(f"Warning: Failed to initialize {backend_name} backend: {e}")
        print("Falling back to aten backend")
        use_backend("aten")


# Auto-configure on import unless explicitly disabled
if os.getenv("BACKENDBENCH_NO_AUTO_PATCH", "").lower() not in ("1", "true", "yes"):
    _auto_configure()
