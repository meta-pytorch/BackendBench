import math
import torch
from torch.testing import make_tensor

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


def _deserialize_tensor(size, dtype, stride=None, device="cuda"):
    kwargs = {}
    if dtype in _FLOATING_TYPES:
        kwargs.update({"low": 0, "high": 1})
<<<<<<< HEAD
<<<<<<< HEAD

    # Fall back to CPU if CUDA is not available
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

=======
    
    # Fall back to CPU if CUDA is not available
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
>>>>>>> 201e39a (Add tests for serialization and deserialization)
=======

    # Fall back to CPU if CUDA is not available
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

>>>>>>> a15dcbc (fix)
    if stride is not None:
        extent = 1 + sum((size - 1) * stride for size, stride in zip(size, stride))
        data = make_tensor(extent, dtype=dtype, device=device, **kwargs)
        return data.as_strided(size, stride)
    return make_tensor(size, dtype=dtype, device=device, **kwargs)

<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 201e39a (Add tests for serialization and deserialization)
=======

>>>>>>> a15dcbc (fix)
def _serialize_tensor(tensor):
    """Helper function to serialize a tensor to string format"""
    shape = list(tensor.shape)
    dtype = dtype_abbrs[tensor.dtype]
    stride = tensor.stride() if not tensor.is_contiguous() else None
<<<<<<< HEAD
<<<<<<< HEAD

=======
    
>>>>>>> 201e39a (Add tests for serialization and deserialization)
=======

>>>>>>> a15dcbc (fix)
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a15dcbc (fix)
    kwargs_parts = [f"'{key}': {_serialize_value(val)}" for key, val in kwargs.items()]

    # Handle empty args tuple properly
    args_str = f"({', '.join(parts)},)" if parts else "()"

    return f"({args_str}, {{{', '.join(kwargs_parts)}}})"
<<<<<<< HEAD
=======
    kwargs_parts = [f"{key}={_serialize_value(val)}" for key, val in kwargs.items()]
    
    return f"(({', '.join(parts)},), {{{', '.join(kwargs_parts)}}})"
=======
>>>>>>> a15dcbc (fix)


<<<<<<< HEAD
# Alias for backward compatibility
reserialize_args = serialize_args
>>>>>>> 201e39a (Add tests for serialization and deserialization)


=======
>>>>>>> 4ed1e55 (fix)
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
    return eval(inps.strip().strip("'").strip('"'), global_vals)
