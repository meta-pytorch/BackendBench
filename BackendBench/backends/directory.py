# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import os
from typing import Callable, Dict

from torch.utils.cpp_extension import load_inline
from ..utils import folder_name_to_op_name, get_pytorch_op
from .base import Backend

logger = logging.getLogger(__name__)


class DirectoryBackend(Backend):
    def __init__(self, ops_dir="generated_kernels"):
        super().__init__("directory")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self._load_kernels()

    def _load_kernels(self):
        """
        Discovers and loads kernel implementations from the operator directory structure.

        This method scans the ops_dir for subdirectories named after PyTorch operator
        overloads (e.g., "add__Tensor" for add.Tensor and "add__Scalar" for add.Scalar).
        Each subdirectory should contain Python files with kernel implementations
        following the naming pattern: {op_name}_implementation*.py

        This method uses the op overload format (e.g., "add__Tensor" for "add.Tensor") and
        registers the kernel for ONLY that specific overload.
        """
        if not os.path.exists(self.ops_dir):
            logger.warning(f"ops directory {self.ops_dir} does not exist")
            return

        loaded_count = 0
        for folder_name in os.listdir(self.ops_dir):
            op_dir = os.path.join(self.ops_dir, folder_name)
            if not os.path.isdir(op_dir):
                continue

            impl_files = [
                f
                for f in os.listdir(op_dir)
                if (f.endswith(".py") or f.endswith(".cu") or f.endswith(".cpp"))
                and f.startswith(f"{folder_name}_implementation")
            ]
            if not impl_files:
                logger.debug(f"No implementation files found in {op_dir}")
                continue

            impl_file = sorted(impl_files)[0]
            impl_path = os.path.join(op_dir, impl_file)

            try:
                op_name = folder_name_to_op_name(folder_name)
                kernel_func = self._load_kernel_from_file(impl_path, op_name)

                pytorch_op = get_pytorch_op(op_name)
                if pytorch_op:
                    self.compiled_kernels[pytorch_op] = kernel_func
                    logger.info(f"Loaded {op_name} from {impl_file} -> {op_name}")
                    loaded_count += 1

            except Exception as e:
                logger.error(f"Error loading {op_name} from {impl_file}: {e}")

        logger.info(f"DirectoryBackend loaded {loaded_count} kernels from {self.ops_dir}/")

    def _load_python_kernel(self, file_path: str, op_name: str) -> Callable:
        """
        Load a kernel implementation from a Python file.

        Args:
            file_path: Path to the Python implementation file
            op_name: Base name of the operator (e.g., "add", "mul", "conv2d")

        Returns:
            Callable kernel implementation function

        Raises:
            ValueError: If the expected kernel function is not found in the file
        """
        spec = importlib.util.spec_from_file_location(f"op_{folder_name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel_func_name = f"{folder_name}_kernel_impl"
        if hasattr(module, kernel_func_name):
            return getattr(module, kernel_func_name)
        else:
            raise ValueError(f"No function named {kernel_func_name} found in {file_path}")

    def setup_cpp_extension(self):
        global CPP_EXTENSION_IS_SETUP
        if not CPP_EXTENSION_IS_SETUP:
            from setuptools import setup
            from torch.utils import cpp_extension

            setup(
                name="extension_cpp",
                ext_modules=[
                    cpp_extension.CppExtension(
                        "extension_cpp",
                        ["muladd.cpp"],
                        # define Py_LIMITED_API with min version 3.9 to expose only the stable
                        # limited API subset from Python.h
                        extra_compile_args={"cxx": ["-DPy_LIMITED_API=0x03090000"]},
                        py_limited_api=True,
                    )
                ],  # Build 1 wheel across multiple Python versions
                cmdclass={"build_ext": cpp_extension.BuildExtension},
                options={
                    "bdist_wheel": {"py_limited_api": "cp39"}
                },  # 3.9 is minimum supported Python version
            )
            CPP_EXTENSION_IS_SETUP = True

    def _load_cuda_kernel(self, file_path: str, op_name: str) -> Callable:
        """
        Load and compile a kernel implementation from CUDA files using load_inline.

        Args:
            file_path: Path to the CUDA implementation file (.cu or .cpp)
            op_name: Base name of the operator (e.g., "add__Tensor")

        Returns:
            Callable kernel implementation function

        Raises:
            ValueError: If the expected kernel function is not found in the compiled module
        """
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        base_name = file_name.rsplit(".", 1)[0]

        cu_file = os.path.join(file_dir, f"{base_name}.cu")
        cpp_file = os.path.join(file_dir, f"{base_name}.cpp")

        cpp_source = ""
        cuda_source = ""

        # Read both files if they exist
        if os.path.exists(cu_file):
            with open(cu_file, "r") as f:
                cuda_source = f.read()
        print(f"cuda_source: {cuda_source}")

        if os.path.exists(cpp_file):
            with open(cpp_file, "r") as f:
                cpp_source = f.read()
        print(f"cpp_source: {cpp_source}")

        # Use load_inline for all cases
        module_name = f"{op_name}_cuda_inline"
        cuda_module = load_inline(
            name=module_name,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=[op_name],
            verbose=True,
        )

        # x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
        # y = torch.randn(4, 4, device="cuda", dtype=torch.float32)

        # print(getattr(cuda_module, op_name)(x, y))

        if hasattr(cuda_module, op_name):
            return getattr(cuda_module, op_name)
        else:
            raise ValueError(
                f"No function named {op_name} found in compiled CUDA module from {file_path}"
            )

    def _load_kernel_from_file(self, file_path: str, op_name: str) -> Callable:
        """
        Dynamically load a kernel implementation function from a Python or CUDA file.

        Dispatches to the appropriate loader based on file extension:
        - .py files -> _load_python_kernel
        - .cu or .cpp files -> _load_cuda_kernel

        Args:
            file_path: Path to the implementation file (Python or CUDA)
            op_name: Base name of the operator (e.g., "add", "mul", "conv2d")

        Returns:
            Callable kernel implementation function

        Raises:
            ValueError: If the file extension is unsupported or the kernel function is not found
        """
        file_ext = os.path.splitext(file_path)[1]

        if file_ext == ".py":
            return self._load_python_kernel(file_path, op_name)
        elif file_ext in [".cu", ".cpp"]:
            return self._load_cuda_kernel(file_path, op_name)
        else:
            raise ValueError(
                f"Unsupported file extension {file_ext} for {file_path}. Expected .py, .cu, or .cpp"
            )

    def __getitem__(self, key):
        if key in self.compiled_kernels:
            return self.compiled_kernels[key]
        raise KeyError(
            f"Operator {key} not implemented in DirectoryBackend - add implementation to {self.ops_dir}/"
        )

    def __contains__(self, key):
        return key in self.compiled_kernels
