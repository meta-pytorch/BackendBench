# kernel.py
# Triton implementation of aten.bitwise_and.Tensor with broadcasting support.
# The actual computation is performed inside the Triton kernel using tl.load/tl.store,
# and a thin Python wrapper named `kernel_function` handles dtype promotion,
# broadcasting, and kernel launch.

import torch
import triton
import triton.language as tl


# Autotune configurations for a memory-bound elementwise op
_configs = [
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
]


@triton.autotune(configs=_configs, key=["n_elements"])
@triton.jit
def _bitwise_and_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements,                               # total number of elements in output
    # Output shape dims (padded to 8 dims, row-major: s0 ... s7, s7 is innermost)
    s0, s1, s2, s3, s4, s5, s6, s7,
    # Strides for a (in elements), padded to 8 dims
    as0, as1, as2, as3, as4, as5, as6, as7,
    # Strides for b (in elements), padded to 8 dims
    bs0, bs1, bs2, bs3, bs4, bs5, bs6, bs7,
    NDIMS: tl.constexpr,                      # actual number of dims (<= 8)
    BLOCK_SIZE: tl.constexpr,                 # kernel tile/block size
):
    # 1D program over output elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    offsets = offsets.to(tl.int64)
    # Mask to guard out-of-bounds
    mask = offsets < n_elements

    # Compute per-element offsets for A and B using row-major unraveling
    rem = offsets
    off_a = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    off_b = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Work from innermost (s7) to outermost (s0).
    # Note: output.numel() > 0 guarantees no dimension size is zero, so division is safe.
    if NDIMS >= 1:
        size = s7
        idx = rem % size
        rem = rem // size
        off_a += idx * as7
        off_b += idx * bs7
    if NDIMS >= 2:
        size = s6
        idx = rem % size
        rem = rem // size
        off_a += idx * as6
        off_b += idx * bs6
    if NDIMS >= 3:
        size = s5
        idx = rem % size
        rem = rem // size
        off_a += idx * as5
        off_b += idx * bs5
    if NDIMS >= 4:
        size = s4
        idx = rem % size
        rem = rem // size
        off_a += idx * as4
        off_b += idx * bs4
    if NDIMS >= 5:
        size = s3
        idx = rem % size
        rem = rem // size
        off_a += idx * as3
        off_b += idx * bs3
    if NDIMS >= 6:
        size = s2
        idx = rem % size
        rem = rem // size
        off_a += idx * as2
        off_b += idx * bs2
    if NDIMS >= 7:
        size = s1
        idx = rem % size
        rem = rem // size
        off_a += idx * as1
        off_b += idx * bs1
    if NDIMS >= 8:
        size = s0
        idx = rem % size
        rem = rem // size
        off_a += idx * as0
        off_b += idx * bs0

    # Load, compute, store. Use masks for out-of-bounds protection.
    a = tl.load(a_ptr + off_a, mask=mask, other=0)
    b = tl.load(b_ptr + off_b, mask=mask, other=0)
    c = a & b
    tl.store(out_ptr + offsets, c, mask=mask)


def _check_supported_dtypes(a: torch.Tensor, b: torch.Tensor):
    if a.dtype.is_floating_point or b.dtype.is_floating_point:
        raise TypeError("bitwise_and only supports boolean and integer dtypes.")
    if (a.dtype == torch.bool) ^ (b.dtype == torch.bool):
        # PyTorch does not allow mixing bool with non-bool
        raise TypeError("bitwise_and does not support mixing bool with non-bool tensors.")


def _promote_dtype(a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
    # PyTorch semantics:
    # - bool & bool -> bool
    # - int & int -> integer type promotion (torch.result_type)
    if a.dtype == torch.bool and b.dtype == torch.bool:
        return torch.bool
    return torch.result_type(a, b)


def _pad_to_8_dims(shape_or_strides):
    # Pads a tuple/list to 8 dims by pre-pending ones/zeros accordingly
    t = tuple(shape_or_strides)
    if len(t) > 8:
        raise ValueError("This kernel currently supports up to 8 dimensions.")
    pad = 8 - len(t)
    # For shapes, pad with 1; for strides, pad with 0 is also OK because those dims won't be used.
    # However, we specifically pad shapes with 1 and strides with 0 via callers.
    return (1,) * pad + t


def bitwise_and_kernel_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise bitwise_and with broadcasting using Triton.

    Args:
        a: input tensor (boolean or integer dtype), CUDA
        b: input tensor (boolean or integer dtype), CUDA

    Returns:
        out: tensor equal to torch.bitwise_and(a, b) with broadcasting.
    """
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("kernel_function expects PyTorch tensors as inputs.")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise RuntimeError("Inputs must be CUDA tensors.")
    if a.device != b.device:
        raise RuntimeError("Inputs must be on the same CUDA device.")

    _check_supported_dtypes(a, b)
    out_dtype = _promote_dtype(a, b)

    # Cast to common dtype as per PyTorch semantics
    if a.dtype != out_dtype:
        a = a.to(out_dtype)
    if b.dtype != out_dtype:
        b = b.to(out_dtype)

    # Compute broadcasted output shape
    out_shape = torch.broadcast_shapes(a.shape, b.shape)

    # Handle zero-sized outputs early
    out = torch.empty(out_shape, device=a.device, dtype=out_dtype)
    if out.numel() == 0:
        return out

    # Expand inputs for broadcasting; this introduces stride=0 where needed
    a_view = a.expand(out_shape)
    b_view = b.expand(out_shape)

    # Prepare shape and strides (pad to 8 dims, row-major: s0 ... s7)
    shape_padded = _pad_to_8_dims(out_shape)
    a_strides = _pad_to_8_dims(a_view.stride())
    b_strides = _pad_to_8_dims(b_view.stride())

    # Kernel launch: 1D grid
    n_elements = out.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch Triton kernel. Note: we do NOT pass storage offsets; pointers already
    # point at the first logical element of each expanded view.
    _bitwise_and_kernel[grid](
        a_view, b_view, out,
        n_elements,
        # shapes (s0..s7)
        shape_padded[0], shape_padded[1], shape_padded[2], shape_padded[3],
        shape_padded[4], shape_padded[5], shape_padded[6], shape_padded[7],
        # a strides (as0..as7)
        a_strides[0], a_strides[1], a_strides[2], a_strides[3],
        a_strides[4], a_strides[5], a_strides[6], a_strides[7],
        # b strides (bs0..bs7)
        b_strides[0], b_strides[1], b_strides[2], b_strides[3],
        b_strides[4], b_strides[5], b_strides[6], b_strides[7],
        NDIMS=len(out_shape),
    )
    return out