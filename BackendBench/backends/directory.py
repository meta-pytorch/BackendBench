# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import os
from typing import Callable, Dict

import torch
import torch.nn.functional

from .base import Backend

logger = logging.getLogger(__name__)


class DirectoryBackend(Backend):
    def __init__(self, ops_dir="generated_kernels"):
        super().__init__("directory")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self.original_ops: Dict[str, Callable] = {}
        self._patched = False
        self._load_kernels()
        self.ops = self.compiled_kernels

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

            # Load the implementation and map to PyTorch operation
            kernel_func = self._load_kernel_from_file(impl_path, op_name)
            pytorch_op = self._find_pytorch_op(op_name)
            if pytorch_op and kernel_func:
                self.compiled_kernels[pytorch_op] = kernel_func
                logger.info(f"Loaded {op_name} from {impl_file}")
                loaded_count += 1
            else:
                logger.warning(f"Could not map {op_name} to PyTorch operation")

        logger.info(f"DirectoryBackend loaded {loaded_count} kernels from {self.ops_dir}/")

    def _load_kernel_from_file(self, file_path: str, op_name: str) -> Callable:
        spec = importlib.util.spec_from_file_location(f"op_{op_name}", file_path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel_func_name = f"{op_name}_kernel_impl"
        return getattr(module, kernel_func_name, None)

    def _find_pytorch_op(self, op_name: str):
        """Map operation name to PyTorch operation."""
        # Try common patterns - prioritize Tensor overload for tensor operations
        op = getattr(torch.ops.aten, op_name, None)
        if not op:
            return None

        # Try Tensor overload first, then Scalar, then default
        for overload in ["Tensor", "Scalar", "default"]:
            if hasattr(op, overload):
                return getattr(op, overload)

        return None

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        # Fallback to original operation if not implemented
        return key

    def __contains__(self, key):
        return key in self.compiled_kernels or True  # Always claim to contain ops for fallback

    def patch_operations(self):
        """Monkey patch PyTorch operations with directory implementations."""
        if self._patched:
            return

        patched_count = 0
        for torch_op, kernel_impl in self.compiled_kernels.items():
            # Store the original __call__ method for ops
            self.original_ops[torch_op] = torch_op.__call__

            # Create a wrapper that calls our implementation
            def make_wrapper(impl):
                def wrapper(*args, **kwargs):
                    return impl(*args, **kwargs)

                return wrapper

            # Replace the __call__ method
            torch_op.__call__ = make_wrapper(kernel_impl)
            patched_count += 1

            # Also patch the corresponding torch function and tensor methods
            self._patch_torch_functions(torch_op, kernel_impl)

        self._patched = True
        logger.info(f"DirectoryBackend: Monkey patched {patched_count} operations")

    def _patch_torch_functions(self, torch_op, kernel_impl):
        """Patch torch functions and tensor methods that correspond to aten ops."""
        # Extract op name: 'aten::add.Tensor' -> 'add'
        op_name = (
            torch_op._name.split("::")[1].split(".")[0]
            if "::" in torch_op._name
            else torch_op._name.split(".")[0]
        )

        # Generate dynamic mappings for this operation
        patch_mappings = self._generate_torch_function_mappings(op_name)

        for target_obj, attr_name in patch_mappings:
            if hasattr(target_obj, attr_name):
                original_func = getattr(target_obj, attr_name)
                # Store original for restoration
                if (target_obj, attr_name) not in self.original_ops:
                    self.original_ops[(target_obj, attr_name)] = original_func

                # Create wrapper with explicit parameter to capture closure correctly
                def make_func_wrapper(impl, name):
                    def wrapper(*args, **kwargs):
                        return impl(*args, **kwargs)

                    wrapper.__name__ = f"patched_{name}"
                    return wrapper

                # Patch the function/method
                wrapped_func = make_func_wrapper(kernel_impl, attr_name)
                setattr(target_obj, attr_name, wrapped_func)

    def _generate_torch_function_mappings(self, op_name):
        """Generate dynamic mappings from aten op name to torch functions/methods."""
        import torch.nn.functional as F
        
        # Special cases for irregular mappings (based on FlagGems patterns)
        special_operator_mappings = {
            "div": "__truediv__",  # div maps to / operator
            "floor_divide": "__floordiv__",  # // operator
            "mod": "__mod__",  # % operator  
            "pow": "__pow__",  # ** operator
            "matmul": "__matmul__",  # @ operator
            "eq": "__eq__",  # == operator
            "ne": "__ne__",  # != operator
            "lt": "__lt__",  # < operator
            "le": "__le__",  # <= operator
            "gt": "__gt__",  # > operator
            "ge": "__ge__",  # >= operator
            "and": "__and__",  # & operator
            "or": "__or__",  # | operator
            "xor": "__xor__",  # ^ operator
            "lshift": "__lshift__",  # << operator
            "rshift": "__rshift__",  # >> operator
            "neg": "__neg__",  # unary - operator
            "pos": "__pos__",  # unary + operator
            "invert": "__invert__",  # ~ operator
            "iadd": "__iadd__",  # += operator
            "isub": "__isub__",  # -= operator
            "imul": "__imul__",  # *= operator
            "itruediv": "__itruediv__",  # /= operator
            "ifloordiv": "__ifloordiv__",  # //= operator
            "imod": "__imod__",  # %= operator
            "ipow": "__ipow__",  # **= operator
            "iand": "__iand__",  # &= operator
            "ior": "__ior__",  # |= operator
            "ixor": "__ixor__",  # ^= operator
            "ilshift": "__ilshift__",  # <<= operator
            "irshift": "__irshift__",  # >>= operator
        }
        
        # Special namespaces for certain operations
        functional_ops = {
            "relu", "gelu", "silu", "mish", "softmax", "log_softmax", "softplus", 
            "softsign", "tanh", "sigmoid", "hardsigmoid", "hardtanh", "hardswish",
            "leaky_relu", "elu", "selu", "celu", "glu", "logsigmoid", "softshrink",
            "hardshrink", "tanhshrink", "threshold", "dropout", "dropout2d", "dropout3d",
            "alpha_dropout", "feature_alpha_dropout", "batch_norm", "instance_norm",
            "group_norm", "layer_norm", "local_response_norm", "normalize", "conv1d",
            "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
            "linear", "bilinear", "embedding", "embedding_bag", "one_hot", "cross_entropy",
            "nll_loss", "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss",
            "max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d",
            "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
            "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
            "interpolate", "upsample", "grid_sample", "affine_grid", "pad",
        }
        
        mappings = []
        
        # 1. Try torch.{op_name} (highest priority for most ops)
        if hasattr(torch, op_name):
            mappings.append((torch, op_name))
            
        # 2. Try torch.Tensor.{op_name} (tensor methods)
        if hasattr(torch.Tensor, op_name):
            mappings.append((torch.Tensor, op_name))
            
        # 3. Try standard operator overload torch.Tensor.__{op_name}__
        standard_dunder = f"__{op_name}__"
        if hasattr(torch.Tensor, standard_dunder):
            mappings.append((torch.Tensor, standard_dunder))
            
        # 4. Try special operator mappings
        if op_name in special_operator_mappings:
            special_dunder = special_operator_mappings[op_name]
            if hasattr(torch.Tensor, special_dunder):
                mappings.append((torch.Tensor, special_dunder))
            
        # 5. Try torch.nn.functional.{op_name} for functional operations
        if op_name in functional_ops and hasattr(F, op_name):
            mappings.append((F, op_name))
            
        return mappings

    def unpatch_operations(self):
        """Restore original PyTorch operations."""
        if not self._patched:
            return

        for key, original_func in self.original_ops.items():
            if isinstance(key, tuple):
                # This is a (target_obj, attr_name) tuple for torch functions/methods
                target_obj, attr_name = key
                setattr(target_obj, attr_name, original_func)
            else:
                # This is a torch_op for aten operations
                key.__call__ = original_func

        self.original_ops.clear()
        self._patched = False
        logger.info("DirectoryBackend: Restored original PyTorch operations")


# Global state for easy monkey patching
_global_backend = None


def globally_override_all_pytorch_ops(ops_dir="generated_kernels"):
    """
    Globally monkey patch all PyTorch operations with custom implementations.

    Args:
        ops_dir: Directory containing custom operator implementations

    Returns:
        DirectoryBackend: The backend instance for manual control if needed
    """
    global _global_backend

    if _global_backend is not None:
        logger.warning(
            "PyTorch operations already globally overridden. Call globally_restore_pytorch_ops() first."
        )
        return _global_backend

    _global_backend = DirectoryBackend(ops_dir)
    _global_backend.patch_operations()
    return _global_backend


def globally_restore_pytorch_ops():
    """
    Restore original PyTorch operations, undoing the global override.
    """
    global _global_backend

    if _global_backend is None:
        logger.warning("No global PyTorch override active.")
        return

    _global_backend.unpatch_operations()
    _global_backend = None


def get_global_backend():
    """
    Get the current global backend instance, if any.

    Returns:
        DirectoryBackend or None: The active global backend
    """
    return _global_backend
