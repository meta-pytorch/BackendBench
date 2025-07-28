import importlib.util
import logging
import os
from typing import Callable, Dict

import torch

from .base import Backend

logger = logging.getLogger(__name__)


class DirectoryBackend(Backend):
    def __init__(self, ops_dir="generated_kernels"):
        super().__init__("directory")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self._load_kernels()

    def _load_kernels(self):
        if not os.path.exists(self.ops_dir):
            logger.warning(f"ops directory {self.ops_dir} does not exist")
            return

        loaded_count = 0
        for op_name in os.listdir(self.ops_dir):
            op_dir = os.path.join(self.ops_dir, op_name)
            if not os.path.isdir(op_dir):
                continue

            impl_files = [f for f in os.listdir(op_dir) if f.endswith(".py")]
            if not impl_files:
                logger.warning(f"No Python files found in {op_dir}")
                continue

            # Use the first implementation file
            impl_file = impl_files[0]
            impl_path = os.path.join(op_dir, impl_file)

            try:
                # Load the implementation and map to PyTorch operation
                kernel_func = self._load_kernel_from_file(impl_path, op_name)
                pytorch_op = self._find_pytorch_op(op_name)
                if pytorch_op:
                    self.compiled_kernels[pytorch_op] = kernel_func
                    logger.info(f"Loaded {op_name} from {impl_file}")
                    loaded_count += 1
                else:
                    logger.warning(f"Could not map {op_name} to PyTorch operation")

            except Exception as e:
                logger.error(f"Error loading {op_name} from {impl_file}: {e}")

        logger.info(f"DirectoryBackend loaded {loaded_count} kernels from {self.ops_dir}/")

    def _load_kernel_from_file(self, file_path: str, op_name: str) -> Callable:
        spec = importlib.util.spec_from_file_location(f"op_{op_name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel_func_name = f"{op_name}_kernel_impl"
        if hasattr(module, kernel_func_name):
            return getattr(module, kernel_func_name)
        else:
            raise ValueError(f"No callable function found in {file_path}")

    def _find_pytorch_op(self, op_name: str):
        """Map operation name to PyTorch operation."""
        # Try common patterns
        try:
            return getattr(torch.ops.aten, op_name).default
        except AttributeError:
            pass

        try:
            return getattr(torch.ops.aten, op_name).Tensor
        except AttributeError:
            pass

        # Not 100% sure this is right, will need to iterate over all ops
        return None

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        # Fallback to original operation if not implemented
        return key

    def __contains__(self, key):
        return key in self.compiled_kernels or True  # Always claim to contain ops for fallback
