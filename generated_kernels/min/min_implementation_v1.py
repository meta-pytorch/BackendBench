# kernel.py
import torch
import triton
import triton.language as tl

"""
Global minimum reduction using Triton, equivalent to aten.min.default (torch.min(input)).

- Handles arbitrary ranks and non-contiguous tensors via explicit shape/stride indexing.
- Supports integer, floating, and bool dtypes.
- Floating dtypes propagate NaNs exactly like PyTorch: if any NaN exists, the result is NaN.
- Reduction is done entirely in Triton using tl.load/tl.store and tl.min/tl.max.

Entry point: kernel_function(x: torch.Tensor) -> torch.Tensor (0-dim scalar with same dtype/device as x)
"""

MAX_DIMS = 6


@triton.jit
def _reduce_min_stage1_general_nd(
    in_ptr,               # *T_in
    out_vals_ptr,         # *T_out (same as input dtype except: bool -> int32)
    out_flags_ptr,        # *uint8 (block has NaN? 1 : 0)
    n_elements,           # int32
    # shapes [0..MAX_DIMS-1]
    s0, s1, s2, s3, s4, s5,
    # strides [0..MAX_DIMS-1] (in elements)
    st0, st1, st2, st3, st4, st5,
    other_init,           # masked load init: +inf for floats, max for ints, True for bool
    NDIMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_IS_FLOAT: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Convert flat offsets -> n-d indices (row-major), then -> addresses using strides
    tmp = offsets.to(tl.int64)
    addr = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    if NDIMS > 5:
        i5 = tmp % s5
        tmp = tmp // s5
        addr += i5 * st5
    if NDIMS > 4:
        i4 = tmp % s4
        tmp = tmp // s4
        addr += i4 * st4
    if NDIMS > 3:
        i3 = tmp % s3
        tmp = tmp // s3
        addr += i3 * st3
    if NDIMS > 2:
        i2 = tmp % s2
        tmp = tmp // s2
        addr += i2 * st2
    if NDIMS > 1:
        i1 = tmp % s1
        tmp = tmp // s1
        addr += i1 * st1
    if NDIMS > 0:
        i0 = tmp % s0
        addr += i0 * st0

    ptrs = in_ptr + addr

    if IS_BOOL:
        # Load bool; masked with True to not affect min
        vals_b = tl.load(ptrs, mask=mask, other=other_init)
        vals_i32 = vals_b.to(tl.int32)
        part_min = tl.min(vals_i32, axis=0)
        tl.store(out_vals_ptr + pid, part_min)
        # No NaN for bool
        tl.store(out_flags_ptr + pid, 0)
    else:
        vals = tl.load(ptrs, mask=mask, other=other_init)
        if DTYPE_IS_FLOAT:
            # NaN detection: NaN != NaN
            nan_mask = (vals != vals) & mask
            # Replace NaNs by +inf (other_init) for numeric min
            clean_vals = tl.where(nan_mask, other_init, vals)
            part_min = tl.min(clean_vals, axis=0)
            # has_nan = any(nan_mask)
            has_nan = tl.max(nan_mask.to(tl.uint8), axis=0)
            tl.store(out_flags_ptr + pid, has_nan)
        else:
            # Integers
            part_min = tl.min(vals, axis=0)
            tl.store(out_flags_ptr + pid, 0)
        tl.store(out_vals_ptr + pid, part_min)


@triton.jit
def _reduce_min_1d_contig(
    x_ptr,           # *T
    y_ptr,           # *T
    n_elements,      # int32
    other_init,      # T
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=other_init)
    m = tl.min(x, axis=0)
    tl.store(y_ptr + pid, m)


@triton.jit
def _reduce_max_uint8_1d_contig(
    x_ptr,           # *uint8
    y_ptr,           # *uint8
    n_elements,      # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    m = tl.max(x, axis=0)
    tl.store(y_ptr + pid, m)


def _ceil_div(a, b):
    return (a + b - 1) // b


def min_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Compute global minimum of x using Triton (equivalent to torch.min(x)).
    Returns a 0-dim tensor with same dtype/device as x.
    """
    if not x.is_cuda:
        raise ValueError("Input must be on CUDA device.")
    if x.numel() == 0:
        raise RuntimeError("min(): cannot operate on an empty tensor")

    device = x.device
    dtype = x.dtype
    is_float = x.is_floating_point()
    is_bool = (dtype == torch.bool)

    # Shapes/strides in elements
    if x.ndim == 0:
        shapes = [1]
        strides = [0]
    else:
        shapes = list(x.shape)
        strides = list(x.stride())

    # Pad to MAX_DIMS
    shapes = (shapes + [1] * MAX_DIMS)[:MAX_DIMS]
    strides = (strides + [0] * MAX_DIMS)[:MAX_DIMS]
    NDIMS = max(1, x.ndim)

    # Init values for masked loads
    if is_float:
        other_init = float("inf")
        partial_dtype = dtype
    elif is_bool:
        other_init = True
        partial_dtype = torch.int32  # reduce bool as int32 {0,1}
    else:
        iinfo = torch.iinfo(dtype)
        other_init = int(iinfo.max)
        partial_dtype = dtype

    BLOCK_SIZE = 1024
    n_elements = x.numel()
    n_blocks = _ceil_div(n_elements, BLOCK_SIZE)

    # Allocate partials and NaN flags
    partial_vals = torch.empty((n_blocks,), device=device, dtype=partial_dtype)
    nan_flags = torch.empty((n_blocks,), device=device, dtype=torch.uint8)

    # Stage 1: general ND reduction to partial minima + NaN flags
    _reduce_min_stage1_general_nd[(n_blocks,)](
        x,
        partial_vals,
        nan_flags,
        n_elements,  # Triton will treat as int32 scalar
        # shapes
        shapes[0], shapes[1], shapes[2], shapes[3], shapes[4], shapes[5],
        # strides (elements)
        strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
        other_init,
        NDIMS=NDIMS,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE_IS_FLOAT=is_float,
        IS_BOOL=is_bool,
    )

    # Reduce partial minima (1D, contiguous) until single value
    curr_vals = partial_vals
    curr_len = curr_vals.numel()
    while curr_len > 1:
        next_len = _ceil_div(curr_len, BLOCK_SIZE)
        next_vals = torch.empty((next_len,), device=device, dtype=curr_vals.dtype)
        _reduce_min_1d_contig[(next_len,)](
            curr_vals, next_vals, curr_len, other_init, BLOCK_SIZE=BLOCK_SIZE
        )
        curr_vals = next_vals
        curr_len = next_len

    # Reduce NaN flags (uint8) via max (any)
    curr_flags = nan_flags
    curr_len_f = curr_flags.numel()
    while curr_len_f > 1:
        next_len_f = _ceil_div(curr_len_f, BLOCK_SIZE)
        next_flags = torch.empty((next_len_f,), device=device, dtype=torch.uint8)
        _reduce_max_uint8_1d_contig[(next_len_f,)](
            curr_flags, next_flags, curr_len_f, BLOCK_SIZE=BLOCK_SIZE
        )
        curr_flags = next_flags
        curr_len_f = next_len_f

    out = torch.empty((), device=device, dtype=dtype)

    if is_float:
        has_nan = bool(int(curr_flags.item()) != 0)
        if has_nan:
            out.fill_(float('nan'))
            return out

    # Write final min
    if is_bool:
        val = curr_vals[0].to(torch.bool)
        out.copy_(val)
    else:
        if curr_vals.dtype != dtype:
            out.copy_(curr_vals[0].to(dtype))
        else:
            out.copy_(curr_vals[0])
    return out