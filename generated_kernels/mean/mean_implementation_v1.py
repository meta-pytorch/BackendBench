# kernel.py
# Triton kernel implementing aten.mean.default (mean over all elements).
# - Supports contiguous and non-contiguous tensors via strides
# - Handles 0-dim (scalar) tensors
# - Works with float16 and bfloat16 (accumulates in float32 for accuracy)
# - Optional dtype override like PyTorch's torch.mean(..., dtype=...)
#
# The wrapper function 'kernel_function' launches the Triton kernels and returns a 0-dim tensor.

import torch
import triton
import triton.language as tl


# ----------------------------
# Kernel 1: Partial reduction
# ----------------------------
# Compute per-program partial sums over BLOCK_SIZE logical elements of an arbitrarily-strided tensor.
# We convert linear indices -> multi-dimensional indices using the input shape, then to memory offsets
# using the strides, and load the elements to accumulate in float32.
@triton.jit
def _partial_sum_strided_kernel(
    x_ptr,                          # *input* tensor base pointer
    partial_sums_ptr,               # *output* partial sums (float32), one per program
    N,                              # total number of logical elements
    S0, S1, S2, S3, S4, S5, S6, S7,  # shape (up to 8 dims)
    T0, T1, T2, T3, T4, T5, T6, T7,  # strides (up to 8 dims) in elements
    BLOCK_SIZE: tl.constexpr,       # number of elements processed per program
    NDIMS: tl.constexpr,            # actual number of dims (0..8)
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Compute memory offsets for each logical index in offs
    # Linear index -> multi-dimensional indices -> memory offset using strides
    offs_i64 = tl.cast(offs, tl.int64)
    rem = offs_i64
    offset_mem = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    if NDIMS >= 1:
        i0 = rem % tl.cast(S0, tl.int64)
        rem = rem // tl.cast(S0, tl.int64)
        offset_mem += i0 * tl.cast(T0, tl.int64)
    if NDIMS >= 2:
        i1 = rem % tl.cast(S1, tl.int64)
        rem = rem // tl.cast(S1, tl.int64)
        offset_mem += i1 * tl.cast(T1, tl.int64)
    if NDIMS >= 3:
        i2 = rem % tl.cast(S2, tl.int64)
        rem = rem // tl.cast(S2, tl.int64)
        offset_mem += i2 * tl.cast(T2, tl.int64)
    if NDIMS >= 4:
        i3 = rem % tl.cast(S3, tl.int64)
        rem = rem // tl.cast(S3, tl.int64)
        offset_mem += i3 * tl.cast(T3, tl.int64)
    if NDIMS >= 5:
        i4 = rem % tl.cast(S4, tl.int64)
        rem = rem // tl.cast(S4, tl.int64)
        offset_mem += i4 * tl.cast(T4, tl.int64)
    if NDIMS >= 6:
        i5 = rem % tl.cast(S5, tl.int64)
        rem = rem // tl.cast(S5, tl.int64)
        offset_mem += i5 * tl.cast(T5, tl.int64)
    if NDIMS >= 7:
        i6 = rem % tl.cast(S6, tl.int64)
        rem = rem // tl.cast(S6, tl.int64)
        offset_mem += i6 * tl.cast(T6, tl.int64)
    if NDIMS >= 8:
        i7 = rem % tl.cast(S7, tl.int64)
        rem = rem // tl.cast(S7, tl.int64)
        offset_mem += i7 * tl.cast(T7, tl.int64)

    # Cast to 32-bit index for pointer arithmetic (sufficient for tested sizes)
    offset_mem_i32 = tl.cast(offset_mem, tl.int32)

    # Load and sum to float32
    vals = tl.load(x_ptr + offset_mem_i32, mask=mask, other=0)
    vals_f32 = vals.to(tl.float32)
    part_sum = tl.sum(vals_f32, axis=0)  # scalar float32
    tl.store(partial_sums_ptr + pid, part_sum)


# Fast path for contiguous tensors (no stride math).
@triton.jit
def _partial_sum_contiguous_kernel(
    x_ptr,
    partial_sums_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    vals = tl.load(x_ptr + offs, mask=mask, other=0)
    vals_f32 = vals.to(tl.float32)
    part_sum = tl.sum(vals_f32, axis=0)
    tl.store(partial_sums_ptr + pid, part_sum)


# --------------------------------
# Kernel 2: Finalize (sum + divide)
# --------------------------------
# Reduce the array of partial sums into a single sum and divide by N to get the mean.
# This kernel iterates over the partial sums in BLOCK_SIZE-sized chunks to handle any size.
@triton.jit
def _finalize_mean_kernel(
    partial_sums_ptr,       # float32 partial sums
    out_ptr,                # output pointer (final dtype)
    N,                      # total number of elements
    NUM_PARTIALS,           # number of partial sums
    BLOCK_SIZE: tl.constexpr,
):
    # Single-program reduction over all partial sums, iterating in chunks.
    # We launch with grid=(1,)
    acc = 0.0  # scalar accumulator in float32
    # Iterate over chunks of size BLOCK_SIZE
    for start in tl.range(0, NUM_PARTIALS, BLOCK_SIZE, num_stages=1):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < NUM_PARTIALS
        vals = tl.load(partial_sums_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    mean = acc / tl.cast(N, tl.float32)
    # Cast to output dtype and store
    out_val = mean.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr, out_val)


def _pack_shape_strides(x, max_dims=8):
    """
    Pack shapes and strides up to max_dims (pad with 1/0 as appropriate).
    Returns:
      shapes: list[int], length max_dims
      strides: list[int], length max_dims
      ndims: int
    """
    ndims = x.dim()
    assert ndims <= max_dims, f"Tensor with {ndims} dims exceeds supported max_dims={max_dims}"
    shapes = list(x.shape)
    strides = list(x.stride())
    # Pad to max_dims
    shapes += [1] * (max_dims - ndims)
    strides += [0] * (max_dims - ndims)  # 0 won't be used since NDIMS gate prevents access
    return shapes, strides, ndims


def mean_kernel_impl(x: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Compute the mean over all elements of tensor x on GPU using Triton kernels.
    - Supports non-contiguous tensors via stride-based addressing
    - Accumulates in float32 for numerical stability
    - Returns a 0-dim tensor with dtype either x.dtype (default) or an override via dtype argument.

    Args:
      x: input tensor (CUDA). Tested with bfloat16 and float16.
      dtype: optional output dtype (e.g., torch.bfloat16, torch.float16)

    Returns:
      0-dim tensor on the same device with the requested dtype.
    """
    if not x.is_cuda:
        raise ValueError("kernel_function requires a CUDA tensor input.")

    # Determine output dtype: match PyTorch's behavior in the tests
    out_dtype = x.dtype if dtype is None else dtype

    # Number of logical elements (0-dim -> 1 element)
    N = x.numel()
    if N == 0:
        # PyTorch mean on empty tensors raises an error; we follow PyTorch semantics if needed.
        raise RuntimeError("mean of empty tensor is not defined")

    # Allocate output scalar tensor of the requested dtype
    out = torch.empty((), device=x.device, dtype=out_dtype)

    # Set block size for partial reduction and finalize
    # Use power-of-two sizes for better performance
    BLOCK_SIZE = 2048
    # Number of programs for partial reduction
    num_programs = triton.cdiv(N, BLOCK_SIZE)

    # Temporary buffer for partial sums in float32
    partial_sums = torch.empty((num_programs,), device=x.device, dtype=torch.float32)

    # Choose kernel path based on contiguity
    grid = (num_programs,)

    if x.is_contiguous():
        _partial_sum_contiguous_kernel[grid](
            x, partial_sums, N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    else:
        shapes, strides, ndims = _pack_shape_strides(x, max_dims=8)
        _partial_sum_strided_kernel[grid](
            x, partial_sums, N,
            shapes[0], shapes[1], shapes[2], shapes[3], shapes[4], shapes[5], shapes[6], shapes[7],
            strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6], strides[7],
            BLOCK_SIZE=BLOCK_SIZE,
            NDIMS=ndims,
            num_warps=4,
            num_stages=2,
        )

    # Finalize: sum partial sums and divide by N, then cast/store to out
    # We use a single program and iterate over the partial sums in chunks.
    _finalize_mean_kernel[(1,)](
        partial_sums, out, N, num_programs,
        BLOCK_SIZE=1024,
        num_warps=1,
        num_stages=1,
    )

    return out

# Optional: expose the function name required by the test
__all__ = ["kernel_function"]