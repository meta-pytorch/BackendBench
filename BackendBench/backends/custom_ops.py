# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from .base import Backend

logger = logging.getLogger(__name__)


class CustomOpsBackend(Backend):
    """
    Filesystem-based custom backend.

    Layout:
      ./custom_ops/<op>/<impl>

    - <impl> can be:
        - a .py file exporting `{op}_kernel_impl`
        - a directory containing an entry .py file exporting `{op}_kernel_impl`
        - a .cu CUDA file (optional). If CuPy is available, will JIT with NVRTC
          and wrap a callable for PyTorch tensors.

    Notes:
      - The associated inputs live in ./custom_ops/<op>/gen_input.py
        (consumed by a matching suite, not by the backend itself.)
    """

    def __init__(self, ops_dir: str = "custom_ops"):
        super().__init__("custom_ops")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self._load_all_ops()

    def _load_all_ops(self) -> None:
        root = Path(self.ops_dir)
        if not root.exists():
            logger.warning(f"ops directory {self.ops_dir} does not exist")
            return

        loaded_count = 0
        for op_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
            op_name = op_dir.name
            try:
                kernel = self._load_one_op(op_dir, op_name)
                if kernel is not None:
                    # Map by op_name; suite will use op_name or a function with __name__==op_name
                    self.compiled_kernels[op_name] = kernel
                    logger.info(f"Loaded {op_name} from {op_dir}")
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading {op_name} from {op_dir}: {e}")

        logger.info(f"CustomOpsBackend loaded {loaded_count} kernels from {self.ops_dir}/")

    def _load_one_op(self, op_dir: Path, op_name: str) -> Optional[Callable]:
        """Load first valid implementation within <op_dir>.

        New structure: <op>/<impl_name>/<op_impl_file>
        - any .py file exporting `{op_name}_kernel_impl`
        - any .cu file containing kernel `{op_name}_kernel_impl` (via CuPy NVRTC)
        """
        # Search subdirectories (impl_name) first
        for impl_dir in sorted([d for d in op_dir.iterdir() if d.is_dir()]):
            # .py implementations
            for p in sorted(impl_dir.glob("*.py")):
                if p.name == "gen_input.py":
                    continue
                func = self._load_kernel_from_py(p, op_name)
                if func:
                    # register under composite key op__impl
                    impl_name = impl_dir.name
                    self.compiled_kernels[f"{op_name}__{impl_name}"] = func
                    # do not return; keep scanning others to load all
            # .cu implementations
            for cu in sorted(impl_dir.glob("*.cu")):
                func = self._load_kernel_from_cuda(cu, op_name)
                if func:
                    impl_name = impl_dir.name
                    self.compiled_kernels[f"{op_name}__{impl_name}"] = func
        # If at least one impl loaded, return a dummy to indicate success
        if any(k.startswith(f"{op_name}__") for k in self.compiled_kernels.keys()):
            return lambda *args, **kwargs: None

        # Fallback to flat files (legacy)
        for p in sorted(op_dir.glob("*.py")):
            if p.name == "gen_input.py":
                continue
            func = self._load_kernel_from_py(p, op_name)
            if func:
                self.compiled_kernels[f"{op_name}__default"] = func
                return func
        for cu in sorted(op_dir.glob("*.cu")):
            func = self._load_kernel_from_cuda(cu, op_name)
            if func:
                self.compiled_kernels[f"{op_name}__default"] = func
                return func

        return None

    def _load_kernel_from_py(self, file_path: Path, op_name: str) -> Optional[Callable]:
        spec = importlib.util.spec_from_file_location(f"op_{op_name}", str(file_path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        kernel_func_name = f"{op_name}_kernel_impl"
        if hasattr(module, kernel_func_name):
            return getattr(module, kernel_func_name)
        return None

    def _load_kernel_from_cuda(self, cu_path: Path, op_name: str) -> Optional[Callable]:
        """Try to JIT compile with CuPy RawModule (NVRTC) if available.

        Expected kernel function name inside CUDA source: `{op_name}_kernel_impl`.
        We'll wrap a callable: (input_tensors..., **kwargs) -> output_tensor(s).
        For simplicity, we demonstrate a unary elementwise kernel signature.
        Users can customize wrappers per op as needed.
        """
        try:
            import cupy as cp  # type: ignore
        except Exception:
            logger.warning(f"CuPy not available; skipping CUDA impl at {cu_path}")
            return None

        code = cu_path.read_text()
        try:
            module = cp.RawModule(code=code, options=("--use_fast_math",))
            kernel = module.get_function(f"{op_name}_kernel_impl")
        except Exception as e:
            logger.error(f"Failed to compile CUDA kernel {cu_path}: {e}")
            return None

        def wrapper(*args, **kwargs):
            # Minimal example: assume single input tensor -> single output tensor, same shape
            # Move tensors to GPU with CuPy
            torch_input = args[0]
            assert isinstance(torch_input, torch.Tensor)
            x_cp = cp.asarray(torch_input.detach().contiguous().cpu().numpy())
            y_cp = cp.empty_like(x_cp)
            n = x_cp.size
            threads = 256
            blocks = (n + threads - 1) // threads
            kernel((blocks,), (threads,), (x_cp, y_cp, n))
            y = torch.from_numpy(cp.asnumpy(y_cp))
            return y.to(torch_input.device)

        return wrapper

    def __getitem__(self, key):
        # Handle both string keys and function objects
        if isinstance(key, str):
            if key in self.compiled_kernels:
                return self.compiled_kernels[key]
        elif hasattr(key, "__name__"):
            if key.__name__ in self.compiled_kernels:
                return self.compiled_kernels[key.__name__]
        # Fallback to original operation if not implemented
        return key

    def __contains__(self, key):
        # Handle both string keys and function objects
        if isinstance(key, str):
            return key in self.compiled_kernels
        elif hasattr(key, "__name__"):
            return key.__name__ in self.compiled_kernels
        return False
