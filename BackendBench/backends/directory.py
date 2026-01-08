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
    def __init__(self, ops_dir="generated_kernels", load_cpp_source=False):
        super().__init__("directory")
        self.ops_dir = ops_dir
        self.compiled_kernels: Dict[str, Callable] = {}
        self.load_cpp_source = load_cpp_source
        self._load_kernels(load_cpp_source)

    def _load_kernels(self, load_cpp_source=False):
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
                kernel_func = self._load_kernel_from_file(impl_path, folder_name, load_cpp_source)

                pytorch_op = get_pytorch_op(op_name)
                if pytorch_op:
                    self.compiled_kernels[pytorch_op] = kernel_func
                    logger.info(f"Loaded {op_name} from {impl_file} -> {op_name}")
                    loaded_count += 1

            except Exception as e:
                logger.error(f"Error loading {op_name} from {impl_file}: {e}")

        logger.info(f"DirectoryBackend loaded {loaded_count} kernels from {self.ops_dir}/")

    def _load_python_kernel(self, file_path: str, folder_name: str) -> Callable:
        """
        Load a kernel implementation from a Python file.

        Args:
            file_path: Path to the Python implementation file
            folder_name: Base name of the operator (e.g., "add__Tensor")

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

    def _generate_cpp_source(self, base_name: str, cuda_source: str) -> str:
        """
        Generate C++ source code from a CUDA file.

        Args:
            file_path: Path to the CUDA implementation file (.cu or .cpp)
            folder_name: Base name of the operator (e.g., "add__Tensor")

        Returns:
            str: Generated C++ source code
        """
        output_lines = []
        # Always include the torch extension header
        output_lines.append("#include <torch/extension.h>\n")
        # Find the function signature for the given base_name
        for line in cuda_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("at::Tensor") and base_name in stripped:
                # Remove the function body if present
                signature = stripped.split("{")[0].rstrip()
                # Ensure it ends with a semicolon
                if not signature.endswith(";"):
                    signature += ";"
                output_lines.append(signature + "\n")
                break  # Only one function per file is expected
        return "".join(output_lines)

    def _load_cuda_kernel(
        self, file_path: str, folder_name: str, load_cpp_source: bool = False
    ) -> Callable:
        """
        Load and compile a kernel implementation from CUDA files using load_inline.

        Args:
            file_path: Path to the CUDA implementation file
            folder_name: Base name of the operator (e.g., "add__Tensor")
            load_cpp_source: Whether to also load the corresponding .cpp file. Defaults to False.

        Returns:
            Callable kernel implementation function

        Raises:
            ValueError: If the expected kernel function is not found in the compiled module
        """
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        base_name = file_name.rsplit(".", 1)[0]

        cu_file = os.path.join(file_dir, f"{base_name}.cu")

        cuda_source = ""
        # Read cuda file if exists
        if os.path.exists(cu_file):
            with open(cu_file, "r") as f:
                cuda_source = f.read()

        if cuda_source == "" and not load_cpp_source:
            logger.warning(f"No CUDA source found for {file_path}.")
            return None

        cpp_source = ""
        if load_cpp_source:
            # Read cpp file if exists
            cpp_file = os.path.join(file_dir, f"{base_name}.cpp")
            if os.path.exists(cpp_file):
                with open(cpp_file, "r") as f:
                    cpp_source = f.read()
        else:
            # Generate cpp file from cuda file
            cpp_source = self._generate_cpp_source(folder_name, cuda_source)

        # Use load_inline for all cases
        module_name = f"{folder_name}_cuda_inline"
        cuda_module = load_inline(
            name=module_name,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=[folder_name],
            no_implicit_headers=True,
        )

        if hasattr(cuda_module, folder_name):
            return getattr(cuda_module, folder_name)
        else:
            raise ValueError(
                f"No function named {folder_name} found in compiled CUDA module from {file_path}"
            )

    def _load_kernel_from_file(
        self, file_path: str, folder_name: str, load_cpp_source: bool = False
    ) -> Callable:
        """
        Dynamically load a kernel implementation function from a Python or CUDA file.

        Dispatches to the appropriate loader based on file extension:
        - .py files -> _load_python_kernel
        - .cu or .cpp files -> _load_cuda_kernel

        Args:
            file_path: Path to the implementation file (Python or CUDA)
            op_name: Base name of the operator (e.g., "add", "mul", "conv2d")
            load_cpp_source: Whether to also load the corresponding .cpp file. Defaults to False.

        Returns:
            Callable kernel implementation function

        Raises:
            ValueError: If the file extension is unsupported or the kernel function is not found
        """
        file_ext = os.path.splitext(file_path)[1]

        if file_ext == ".py":
            return self._load_python_kernel(file_path, folder_name)
        elif file_ext in [".cu", ".cpp"]:
            return self._load_cuda_kernel(file_path, folder_name, load_cpp_source)
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
