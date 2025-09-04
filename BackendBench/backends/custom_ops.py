# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional, List

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

    Notes:
      - The associated inputs live in ./custom_ops/<op>/gen_input.py
        (consumed by a matching suite, not by the backend itself.)
    """

    def __init__(self, ops_dir: str = "custom_ops"):
        super().__init__("custom_ops")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        # Implementation importers (order defines priority when multiple match a dir)
        self.impl_loaders: List[BaseImplLoader] = []  # type: ignore[name-defined]
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

        logger.info(f"CustomOpsBackend loaded {loaded_count} ops from {self.ops_dir}/")

    def _load_one_op(self, op_dir: Path, op_name: str) -> Optional[Callable]:
        """Load first valid implementation within <op_dir>.

        New structure: <op>/<impl_name>/<op_impl_file>
        - any .py file exporting `{op_name}_kernel_impl`
        """
        # Search subdirectories (impl_name) first
        for impl_dir in sorted([d for d in op_dir.iterdir() if d.is_dir()]):
            impl_name = impl_dir.name
            loaded = False
            for loader in self.impl_loaders:
                if not loader.enabled:
                    continue
                if not loader.supports(impl_dir):
                    continue
                func = loader.load(op_name, impl_dir)
                if func is not None:
                    self._register_impl(op_name, impl_name, func)
                    loaded = True
                    break  # one importer per impl_dir
            if not loaded:
                logger.debug(f"No supported importer recognized implementation dir: {impl_dir}")
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


    def _register_impl(self, op_name: str, impl_name: str, func: Callable) -> None:
        key = f"{op_name}__{impl_name}"
        self.compiled_kernels[key] = func
        logger.info(f"Registered implementation {key}")

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


class BaseImplLoader:
    name: str = "base"
    enabled: bool = True

    def supports(self, impl_dir: Path) -> bool:
        raise NotImplementedError

    def load(self, op_name: str, impl_dir: Path) -> Optional[Callable]:
        raise NotImplementedError


class PythonImplLoader(BaseImplLoader):
    name = "python"
    enabled = True

    def supports(self, impl_dir: Path) -> bool:
        return any(p.suffix == ".py" for p in impl_dir.iterdir())

    def load(self, op_name: str, impl_dir: Path) -> Optional[Callable]:
        # Prefer <op_name>.py, else first .py containing <op_name>_kernel_impl
        candidates = []
        primary = impl_dir / f"{op_name}.py"
        if primary.exists():
            candidates.append(primary)
        candidates.extend([p for p in sorted(impl_dir.glob("*.py")) if p.name != "gen_input.py"]) 
        for p in candidates:
            try:
                spec = importlib.util.spec_from_file_location(f"op_{op_name}", str(p))
                module = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(module)
                kernel_name = f"{op_name}_kernel_impl"
                if hasattr(module, kernel_name):
                    return getattr(module, kernel_name)
            except Exception as e:
                logger.warning(f"Failed to load python impl from {p}: {e}")
        return None


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


class TritonImplLoader(BaseImplLoader):
    name = "triton"
    enabled = _triton_available() and torch.cuda.is_available()

    def supports(self, impl_dir: Path) -> bool:
        if not self.enabled or "triton" not in impl_dir.name.lower():
            return False
        return any(p.suffix == ".py" for p in impl_dir.iterdir())

    def load(self, op_name: str, impl_dir: Path) -> Optional[Callable]:
        # Prefer <op_name>.py
        primary = impl_dir / f"{op_name}.py"
        candidates = [primary] if primary.exists() else []
        if not candidates:
            candidates = sorted(impl_dir.glob("*.py"))
        for p in candidates:
            try:
                spec = importlib.util.spec_from_file_location(f"op_{op_name}", str(p))
                module = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(module)
                kernel_name = f"{op_name}_kernel_impl"
                if hasattr(module, kernel_name):
                    triton_impl = getattr(module, kernel_name)

                    def _wrapped_triton_impl(*args, **kwargs):
                        if len(args) == 0 or not isinstance(args[0], torch.Tensor):
                            raise ValueError("Triton impl expects first arg as torch.Tensor")
                        x = args[0]
                        alpha = kwargs.get("alpha", 1.0)

                        orig_device = x.device
                        orig_dtype = x.dtype

                        if not x.is_cuda:
                            if not torch.cuda.is_available():
                                raise RuntimeError("CUDA required for Triton impl but not available")
                            x = x.to("cuda")
                        x_kernel = x.contiguous().to(torch.float32)

                        y_kernel = triton_impl(x_kernel, alpha)

                        y = y_kernel.to(orig_dtype)
                        if orig_device.type == "cpu":
                            y = y.to("cpu")
                        return y

                    return _wrapped_triton_impl
            except Exception as e:
                logger.warning(f"Failed to load triton impl from {p}: {e}")
        return None

def _env_summary() -> str:
    return (
        f"env: triton={_triton_available()} cuda={torch.cuda.is_available()}"
    )

def _log_env_once():
    logger.info(_env_summary())

_ENV_LOGGED = False

def _maybe_log_env():
    global _ENV_LOGGED
    if not _ENV_LOGGED:
        _log_env_once()
        _ENV_LOGGED = True

def _enabled_loaders_summary(loaders: List[BaseImplLoader]) -> str:
    return ", ".join([f"{l.name}:{'on' if l.enabled else 'off'}" for l in loaders])

def _log_loaders(loaders: List[BaseImplLoader]):
    logger.info(f"custom ops importers -> {_enabled_loaders_summary(loaders)}")

def _init_impl_loaders(self: "CustomOpsBackend") -> None:
    _maybe_log_env()
    loaders: List[BaseImplLoader] = []
    if _triton_available() and torch.cuda.is_available():
        loaders.append(TritonImplLoader())
    loaders.append(PythonImplLoader())
    _log_loaders(loaders)
    self.impl_loaders = loaders


# Ensure loaders are initialized when module is imported and backend instantiated
_ORIG_INIT = CustomOpsBackend.__init__

def _patched_init(self, ops_dir: str = "custom_ops"):
    _ORIG_INIT(self, ops_dir)
    _init_impl_loaders(self)
    # reload ops after loaders available
    self.compiled_kernels.clear()
    self._load_all_ops()

CustomOpsBackend.__init__ = _patched_init  # type: ignore[assignment]
