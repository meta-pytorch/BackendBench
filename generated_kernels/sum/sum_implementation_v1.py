# kernel.py
import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------
# Utilities
# ---------------------------------------------

def _canonicalize_dims(dims, ndim):
    """Turn dims (list/tuple) into sorted, unique, positive dims in [0, ndim)."""
    if isinstance(dims, int):
        dims = [dims]
    dims = [d + ndim if d < 0 else d for d in dims]
    if any(d < 0 or d >= ndim for d in dims):
        raise ValueError(f"Invalid reduction dim in {dims} for ndim={ndim}")
    # unique while preserving input order
    seen = set()
    dims_unique = []
    for d in dims:
        if d not in seen:
            seen.add(d)
            dims_unique.append(d)
    return dims_unique


def _dtype_code(dtype: torch.dtype) -> int:
    """
    Encode dtype for Triton kernel switch.
    0: float16
    1: bfloat16
    2: int16
    3: int32
    4: int64
    5: float32
    """
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.int16:
        return 2
    if dtype == torch.int32:
        return 3
    if dtype == torch.int64:
        return 4
    if dtype == torch.float32:
        return 5
    raise ValueError(f"Unsupported dtype: {dtype}")


def _is_floating(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16, torch.float32)


def _prod(xs):
    p = 1
    for v in xs:
        p *= int(v)
    return int(p)


def _build_keep_struct(shape, strides, reduce_set):
    """
    Build arrays for kept dims (not reduced), preserving original order.
    Returns:
      keep_dims, keep_shape, keep_strides, keep_cumprod
    keep_cumprod[j] = product of keep_shape[j+1:]
    """
    keep_dims = [i for i in range(len(shape)) if i not in reduce_set]
    keep_shape = [int(shape[d]) for d in keep_dims]
    keep_strides = [int(strides[d]) for d in keep_dims]
    # cumprod (row-major) to decode linear index into multidim indices
    keep_cumprod = []
    running = 1
    for j in range(len(keep_shape) - 1, -1, -1):
        keep_cumprod.append(running)
        running *= int(keep_shape[j])
    keep_cumprod = list(reversed(keep_cumprod))  # now keep_cumprod[j] = product of keep_shape[j+1:]
    return keep_dims, keep_shape, keep_strides, keep_cumprod


def _build_reduce_struct(shape, strides, reduce_dims):
    """
    Build arrays for reduced dims preserving original order, plus a list of all
    linear offsets for all coordinates in the reduced subspace (for pointer arithmetic).
    Returns:
      red_shape, red_strides, red_cumprod, red_offsets
    red_cumprod[j] = product of red_shape[j+1:]
    red_offsets: list length product(red_shape) with base 0 order chosen so that last dim varies fastest.
    """
    red_shape = [int(shape[d]) for d in reduce_dims]
    red_strides = [int(strides[d]) for d in reduce_dims]
    # cumprod
    red_cumprod = []
    running = 1
    for j in range(len(red_shape) - 1, -1, -1):
        red_cumprod.append(running)
        running *= int(red_shape[j])
    red_cumprod = list(reversed(red_cumprod))

    # build offsets linearly: last dimension varies fastest
    red_total = _prod(red_shape)
    red_offsets = []
    if red_total > 0:
        # iterative digits decoding without recursion
        for idx in range(red_total):
            off = 0
            rem = idx
            for j in range(len(red_shape)):
                dim = red_shape[j]
                step = red_cumprod[j]
                coord = 0 if dim == 0 else (rem // step) % dim
                off += coord * red_strides[j]
            red_offsets.append(off)
    return red_shape, red_strides, red_cumprod, red_offsets


# ---------------------------------------------
# Triton kernel
# ---------------------------------------------

@triton.jit
def _sum_reduce_kernel(
    x_ptr,  # *x element type*
    y_ptr,  # *y element type*
    out_numel,  # number of output elements (product of kept dims)
    keep_shape_ptr,  # int64[MAX_DIMS]
    keep_cumprod_ptr,  # int64[MAX_DIMS] (product of dims to the right)
    keep_strides_ptr,  # int64[MAX_DIMS]
    red_offsets_ptr,  # int64[red_total] (each is sum(coord[j] * stride_red[j]))
    red_total,  # total elements in reduced subspace
    MAX_DIMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACC_IS_FLOAT: tl.constexpr,  # True: use fp32 accumulator; False: use int64 accumulator
    OUT_DTYPE_CODE: tl.constexpr,  # see _dtype_code()
):
    """
    Generic N-D sum reduction over a set of dimensions.

    Launch:
      1D grid over output elements, BLOCK_SIZE threads per program.

    Indexing:
      - For each output element (outer_id), decode its coordinates across the kept
        dimensions using keep_cumprod and keep_shape.
      - Compute base pointer offset as sum(coord[j] * keep_strides[j]).
      - Iterate over all reduction offsets red_offsets_ptr to accumulate the sum.

    Dtypes:
      - Floating inputs: accumulate in f32, cast to output dtype at the end.
      - Integer inputs: accumulate in i64, cast to output dtype at the end.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask_o = offs < out_numel

    # Decode kept-dim coordinates and compute base offsets
    # Base offsets computed in element strides (not bytes).
    base_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Using padded arrays of length MAX_DIMS; any extra entries are shape=1, cumprod=1, stride=0.
    for j in tl.static_range(0, MAX_DIMS):
        kshape = tl.load(keep_shape_ptr + j)
        kcp = tl.load(keep_cumprod_ptr + j)
        kstride = tl.load(keep_strides_ptr + j)
        # coord along this kept dimension for each output index
        # coord_j = (offs // kcp) % kshape
        coord_j = (offs // kcp) % kshape
        base_offsets += coord_j.to(tl.int64) * kstride

    # Initialize accumulator
    if ACC_IS_FLOAT:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Accumulate over reduction subspace by iterating over precomputed offsets
    # If red_total == 0, this loop runs 0 times and acc stays zero (sum over empty = 0)
    for r in tl.range(0, red_total):
        roff = tl.load(red_offsets_ptr + r)
        ptrs = x_ptr + (base_offsets + roff)
        # Load input values; masked by output mask (invalid output lanes do nothing)
        val = tl.load(ptrs, mask=mask_o, other=0)
        if ACC_IS_FLOAT:
            val = val.to(tl.float32)
        else:
            # integer path: widen to int64
            # Note: load dtype may be int16/int32/int64; .to(int64) is safe
            val = val.to(tl.int64)
        acc += val

    # Cast accumulator to output dtype and store
    # Manual dtype switch based on OUT_DTYPE_CODE
    if OUT_DTYPE_CODE == 0:
        out_vals = acc.to(tl.float16)
    elif OUT_DTYPE_CODE == 1:
        out_vals = acc.to(tl.bfloat16)
    elif OUT_DTYPE_CODE == 2:
        out_vals = acc.to(tl.int16)
    elif OUT_DTYPE_CODE == 3:
        out_vals = acc.to(tl.int32)
    elif OUT_DTYPE_CODE == 4:
        out_vals = acc.to(tl.int64)
    elif OUT_DTYPE_CODE == 5:
        out_vals = acc.to(tl.float32)
    else:
        # Shouldn't happen; default to float32
        out_vals = acc.to(tl.float32)

    tl.store(y_ptr + offs, out_vals, mask=mask_o)


# ---------------------------------------------
# Public wrapper
# ---------------------------------------------

def sum_kernel_impl(x: torch.Tensor, dims, keepdim: bool, dtype: torch.dtype = None):
    """
    Triton implementation of aten.sum.dim_IntList (sum over specified dimensions).

    Args:
      x: Input tensor (CUDA tensor).
      dims: Dimension or list of dimensions to reduce (can be negative, can be unsorted).
      keepdim: Whether to keep reduced dimensions with size 1.
      dtype: Optional dtype of the output (overrides default behavior).

    Returns:
      y: Output tensor on CUDA, contiguous, with sum reduced over `dims`.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA")

    # Canonicalize dims
    ndim = x.dim()
    dims_list = _canonicalize_dims(dims, ndim)
    reduce_set = set(dims_list)

    # Determine output dtype
    if dtype is None:
        out_dtype = x.dtype
    else:
        out_dtype = dtype

    # Build output shape (match PyTorch behavior)
    if keepdim:
        out_shape = [1 if i in reduce_set else int(x.shape[i]) for i in range(ndim)]
    else:
        out_shape = [int(x.shape[i]) for i in range(ndim) if i not in reduce_set]
        if len(out_shape) == 0:
            # Reduce to scalar (0-dim tensor). We'll materialize as shape [1] then view.
            out_shape = []

    # Prepare strides and shapes (in elements)
    in_shape = list(x.shape)
    in_strides = list(x.stride())

    # Kept dims structures
    keep_dims, keep_shape, keep_strides, keep_cumprod = _build_keep_struct(in_shape, in_strides, reduce_set)
    outer_numel = _prod(keep_shape)  # number of output elements

    # Reduced dims structures and offsets
    red_shape, red_strides, red_cumprod, red_offsets = _build_reduce_struct(in_shape, in_strides, dims_list)
    red_total = len(red_offsets)  # product of reduced dims, or 0 if any reduced dimension is 0

    # Allocate output
    # For empty tensor (no dims), PyTorch returns 0-dim if keepdim=False; handle afterwards.
    if len(out_shape) == 0:
        y = torch.empty((), device=x.device, dtype=out_dtype)
    else:
        y = torch.empty(out_shape, device=x.device, dtype=out_dtype)

    # If there are zero output elements, nothing to do; return empty/zero-sized as is.
    if outer_numel == 0:
        return y

    # Padded arrays for Triton (constant MAX_DIMS)
    # We keep MAX_DIMS modestly high to cover typical tensors; tests go up to 5 dims.
    MAX_DIMS = 8
    def pad_list(lst, pad_value, L=MAX_DIMS):
        lst = list(lst)
        if len(lst) > L:
            raise ValueError(f"Exceeded MAX_DIMS={L} with list of length {len(lst)}")
        return lst + [pad_value] * (L - len(lst))

    keep_shape_pad = torch.tensor(pad_list(keep_shape, 1, MAX_DIMS), device=x.device, dtype=torch.int64)
    keep_cumprod_pad = torch.tensor(pad_list(keep_cumprod, 1, MAX_DIMS), device=x.device, dtype=torch.int64)
    keep_strides_pad = torch.tensor(pad_list(keep_strides, 0, MAX_DIMS), device=x.device, dtype=torch.int64)

    # Reduction offsets buffer (can be zero-sized)
    red_offsets_t = torch.tensor(red_offsets if len(red_offsets) > 0 else [0], device=x.device, dtype=torch.int64)
    if red_total == 0:
        # Ensure it's safe to pass; kernel won't read as loop runs 0 times
        red_offsets_t = red_offsets_t[:0]

    # Decide accumulator type path
    acc_is_float = _is_floating(x.dtype) or _is_floating(out_dtype)
    out_dtype_code = _dtype_code(out_dtype)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(outer_numel, BLOCK_SIZE),)

    # Ensure input and output are contiguous pointers in element indexing space.
    # Triton pointer arithmetic uses element strides (not bytes) because pointer is typed.
    x_c = x
    y_c = y.view(-1) if y.dim() != 0 else y  # flatten for linear indexing

    _sum_reduce_kernel[grid](
        x_c,  # ptr
        y_c,  # ptr
        outer_numel,
        keep_shape_pad,
        keep_cumprod_pad,
        keep_strides_pad,
        red_offsets_t,
        red_total,
        MAX_DIMS=MAX_DIMS,
        BLOCK_SIZE=BLOCK_SIZE,
        ACC_IS_FLOAT=acc_is_float,
        OUT_DTYPE_CODE=out_dtype_code,
    )

    # Reshape output to requested shape
    if y.dim() == 0:
        return y
    else:
        return y.view(out_shape)