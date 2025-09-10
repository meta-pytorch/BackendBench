# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import ast
import gc
import inspect
import logging
import math
import re
import textwrap
import importlib.util

import torch
from torch.testing import make_tensor

from typing import Callable

logger = logging.getLogger(__name__)

dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

dtype_abbrs_parsing = {value: key for key, value in dtype_abbrs.items()}

_FLOATING_TYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def uses_cuda_stream(func) -> bool:
    """
    Detects whether a Python function creates CUDA streams.

    Args:
        func: The Python function to analyze

    Returns:
        bool: True if CUDA streams are created, False otherwise
    """
    try:
        source = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(source))
    except (TypeError, OSError, IndentationError, SyntaxError):
        # Handle builtin functions, OpOverload objects, and other callables
        # without source code. These cannot create CUDA streams.
        # Also if the code doesn't compile, then there is not a CUDA stream as the code just isn't runnable
        return False

    # Check for stream creation patterns
    patterns = [
        r"torch\.cuda\.Stream\(",  # torch.cuda.Stream() constructor
        r"cupy\.cuda\.Stream\(",  # cupy.cuda.Stream() constructor
        r"cuda\.Stream\(",  # Generic cuda.Stream() constructor
        r"pycuda.*Stream\(",  # PyCUDA stream creation
        r"\bStream\(",  # Stream() constructor calls
        r"make_stream\(",  # make_stream() factory function
        r"create_stream\(",  # create_stream() factory function
    ]

    if any(re.search(p, source, re.IGNORECASE) for p in patterns):
        return True

    class StreamCreationFinder(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Call(self, node):
            # Check for Stream() constructor calls
            if hasattr(node.func, "attr") and node.func.attr == "Stream":
                self.found = True
            elif hasattr(node.func, "id") and node.func.id == "Stream":
                self.found = True
            self.generic_visit(node)

    finder = StreamCreationFinder()
    finder.visit(tree)
    return finder.found


def _deserialize_tensor(size, dtype, stride=None, device="cuda"):
    kwargs = {}
    if dtype in _FLOATING_TYPES:
        kwargs.update({"low": 0, "high": 1})

    # Fall back to CPU if CUDA is not available
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if stride is not None:
        extent = 1 + sum((size - 1) * stride for size, stride in zip(size, stride))
        data = make_tensor(extent, dtype=dtype, device=device, **kwargs)
        return data.as_strided(size, stride)
    return make_tensor(size, dtype=dtype, device=device, **kwargs)


def _serialize_tensor(tensor):
    """Helper function to serialize a tensor to string format"""
    shape = list(tensor.shape)
    dtype = dtype_abbrs[tensor.dtype]
    stride = tensor.stride() if not tensor.is_contiguous() else None

    if stride:
        return f"T({shape}, {dtype}, {list(stride)})"
    else:
        return f"T({shape}, {dtype})"


def _serialize_value(value):
    """Helper function to serialize any value (tensor, list, primitive)"""
    if isinstance(value, torch.Tensor):
        return _serialize_tensor(value)
    elif isinstance(value, list):
        list_parts = [_serialize_value(item) for item in value]
        return f"[{', '.join(list_parts)}]"
    else:
        return repr(value)


def serialize_args(args, kwargs) -> str:
    """Convert args and kwargs back to the BackendBench string format

    Args:
        args: List of arguments (can contain tensors, lists, primitives)
        kwargs: Dict of keyword arguments

    Returns:
        Serialized string in format: (arg1, arg2, ..., key1=val1, key2=val2, ...)
    """
    if args is None or kwargs is None:
        return "None"

    # Process positional arguments
    parts = [_serialize_value(arg) for arg in args]

    # Process keyword arguments
    kwargs_parts = [f"'{key}': {_serialize_value(val)}" for key, val in kwargs.items()]

    # Handle empty args tuple properly
    args_str = f"({', '.join(parts)},)" if parts else "()"

    return f"({args_str}, {{{', '.join(kwargs_parts)}}})"


def deserialize_args(inps):
    inps = inps.strip().strip("'")
    global_vals = {
        "T": _deserialize_tensor,
        "th": torch,
        "inf": math.inf,
        "torch": torch,
        **dtype_abbrs_parsing,
    }
    # f strings introduce quotations we dont want
    for key in dtype_abbrs_parsing:
        inps = inps.replace(f"'{key}'", key)

    # Handle torch.device strings - replace "torch.device(...)" with torch.device(...)
    # This regex finds patterns like "torch.device('cpu')" or 'torch.device("cuda:0")'
    pattern = r'["\']torch\.device\((.*?)\)["\']'
    inps = re.sub(pattern, r"torch.device(\1)", inps)

    return eval(inps.strip().strip("'").strip('"'), global_vals)


def compute_errors(ref, res, eps=1e-10):
    """Compute max absolute and relative errors between reference and result tensors.

    Returns:
        Tuple of (max_absolute_error, max_relative_error) or (None, None) if not tensors/list of tensors or we fail to compute errors from ref and res
    """
    if isinstance(ref, torch.Tensor) and isinstance(res, torch.Tensor):
        if ref.shape != res.shape:
            return None, None

        if ref.is_sparse and res.is_sparse:
            # todo: create note that we don't calculate errors for sparse tensors / results
            return None, None

        if ref.numel() == 0 and res.numel() == 0:
            # if both are empty tensors, we consider them equal
            return 0.0, 0.0

        # Convert to float for error calculation
        ref_float = ref.float()
        res_float = res.float()

        # Absolute error
        abs_error = (ref_float - res_float).abs().max().item()

        # Relative error (avoid division by zero)
        ref_abs = ref_float.abs()
        rel_error = ((ref_float - res_float).abs() / (ref_abs + eps)).max().item()

        return abs_error, rel_error
    elif isinstance(ref, (list, tuple)) and isinstance(res, (list, tuple)):
        if len(ref) != len(res):
            return None, None

        # if we have no tensors just return None
        if not any(isinstance(x, torch.Tensor) for x in ref) or not any(
            isinstance(x, torch.Tensor) for x in res
        ):
            return None, None

        # For lists/tuples, compute max error across all elements.
        # We will return the maximum of these maxima
        max_abs_error = -math.inf
        max_rel_error = -math.inf

        for r, s in zip(ref, res):
            abs_err, rel_err = compute_errors(r, s)
            if abs_err is None or rel_err is None:
                continue
            max_abs_error = max(max_abs_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

        if max_abs_error == -math.inf:
            return None, None

        return max_abs_error, max_rel_error
    else:
        return None, None


def cleanup_memory_and_gpu():
    """Helper function to clean up GPU memory"""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def get_pytorch_op(op_name: str):
    """
    Convert an operator name string to the actual PyTorch operation object.

    PyTorch operations are structured as torch.ops.aten.{base_name}.{overload}.
    This method handles the conversion from string names like "add.Tensor" or "relu.default"
    to the actual callable operation objects that can be used for dispatch registration.

    Args:
        op_name: String name like "add.Tensor", "relu.default", or just "relu"

    Returns:
        PyTorch operation object that can be used for dispatch registration,
        or None if the operation doesn't exist in torch.ops.aten
    """
    try:
        if "." in op_name:
            base_name, overload = op_name.split(".", 1)
            if overload == "default":
                return getattr(torch.ops.aten, base_name).default
            else:
                return getattr(getattr(torch.ops.aten, base_name), overload)
        else:
            return getattr(torch.ops.aten, op_name).default
    except AttributeError:
        logger.warning(f"Could not find PyTorch operation for {op_name}")
        return None


def extract_operator_name(op_str: str) -> str:
    """Extract clean operator name from various operator string formats."""
    if "aten." in op_str:
        return op_str.split("aten.")[-1].split(".")[0]
    elif "." in op_str:
        return op_str.split(".")[0]
    else:
        return op_str


def compile_kernel_from_string(
    kernel_code: str, op_name: str, kernel_file_path: str, expected_fn_name: str
) -> tuple[Callable | None, list[str]]:
    def _prepare_triton_code(kernel_code: str) -> str:
        """Prepare Triton kernel code with necessary imports."""
        imports = """
import torch
import triton
import triton.language as tl
"""
        if "import torch" not in kernel_code:
            kernel_code = imports + kernel_code
        return kernel_code

    def _prepare_torch_code(kernel_code: str) -> str:
        """Prepare regular PyTorch kernel code with necessary imports."""
        imports = """
import torch
import torch.nn.functional as F
"""
        if "import torch" not in kernel_code:
            kernel_code = imports + kernel_code
        return kernel_code

    def _find_kernel_function(module, op_name: str) -> Callable:
        """Find the main kernel function in the compiled module."""
        expected_name = f"{op_name}_kernel_impl"

        if hasattr(module, expected_name):
            return getattr(module, expected_name)

        available_functions = [
            name
            for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith("_")
        ]

        raise ValueError(
            f"Expected function '{expected_name}' not found in kernel code for {op_name}. "
            f"Available functions: {available_functions}. "
            f"Please ensure the LLM generated code follows the naming convention: {op_name}_kernel_impl"
        )

    try:
        is_triton = "triton.jit" in kernel_code or "@triton.jit" in kernel_code

        if is_triton:
            full_code = _prepare_triton_code(kernel_code)
        else:
            full_code = _prepare_torch_code(kernel_code)

        with open(kernel_file_path, "w") as f:
            f.write(full_code)

        logger.debug(f"Saved kernel to: {kernel_file_path}")

        spec = importlib.util.spec_from_file_location(
            expected_fn_name, kernel_file_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel_func = _find_kernel_function(module, op_name)
        return kernel_func

    except Exception as e:
        raise RuntimeError(f"Failed to compile kernel for {op_name}: {str(e)}") from e
