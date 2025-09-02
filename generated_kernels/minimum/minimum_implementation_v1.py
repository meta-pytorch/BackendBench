import torch
import triton
import triton.language as tl


@triton.jit
def _minimum_broadcast_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    out_shape_ptr,
    a_strides_ptr, b_strides_ptr,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_KIND: tl.constexpr,  # 0: float, 1: int, 2: bool
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    offs = offs.to(tl.int64)
    mask = offs < n_elements

    # Compute input offsets from flattened indices using broadcasted strides
    curr = offs
    a_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    b_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Decompose flat index into multi-dimensional indices (from last dim to first)
    for i in range(RANK):
        k = RANK - 1 - i
        dim_sz = tl.load(out_shape_ptr + k, mask=True, other=1).to(tl.int64)
        dim_sz = tl.where(dim_sz == 0, 1, dim_sz)  # guard (n_elements > 0 implies no zeros, but safe)
        idx_k = curr % dim_sz
        curr = curr // dim_sz

        sa = tl.load(a_strides_ptr + k, mask=True, other=0).to(tl.int64)
        sb = tl.load(b_strides_ptr + k, mask=True, other=0).to(tl.int64)
        a_off += idx_k * sa
        b_off += idx_k * sb

    # Load inputs
    a_val = tl.load(a_ptr + a_off, mask=mask, other=0)
    b_val = tl.load(b_ptr + b_off, mask=mask, other=0)

    # Compute elementwise minimum with correct semantics
    if DTYPE_KIND == 2:
        # bool: False < True, so min == logical AND
        res = a_val & b_val
    elif DTYPE_KIND == 0:
        # Floating-point: propagate NaNs like torch.minimum
        # Detect NaNs without tl.math.isnan (x != x is True only for NaN)
        a_nan = a_val != a_val
        b_nan = b_val != b_val
        any_nan = a_nan | b_nan
        # Tie-break to 'a' on equality (<=) to be deterministic
        min_nb = tl.where(a_val <= b_val, a_val, b_val)
        # For NaN lanes, produce NaN. a_val + b_val is NaN if either is NaN.
        nan_val = a_val + b_val
        res = tl.where(any_nan, nan_val, min_nb)
    else:
        # Integers: standard comparison; tie-break to 'a' on equality
        res = tl.where(a_val <= b_val, a_val, b_val)

    # Store
    tl.store(out_ptr + offs, res, mask=mask)


def _prepare_broadcast_views(a: torch.Tensor, b: torch.Tensor):
    a_exp, b_exp = torch.broadcast_tensors(a, b)
    out_shape = a_exp.shape
    return a_exp, b_exp, out_shape


def _make_index_tensors(shape, a_strides, b_strides, device):
    shape_t = torch.as_tensor(shape, dtype=torch.int64, device=device)
    a_strides_t = torch.as_tensor(a_strides, dtype=torch.int64, device=device)
    b_strides_t = torch.as_tensor(b_strides, dtype=torch.int64, device=device)
    return shape_t, a_strides_t, b_strides_t


def minimum_kernel_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be CUDA tensors.")
    if a.dtype != b.dtype:
        raise ValueError(f"Inputs must have the same dtype, got {a.dtype} vs {b.dtype}")

    a_exp, b_exp, out_shape = _prepare_broadcast_views(a, b)

    # Handle zero-size outputs early
    if len(out_shape) == 0:
        n_elements = 1
    else:
        n = 1
        for s in out_shape:
            n *= s
        n_elements = n

    if n_elements == 0:
        return torch.empty(out_shape, dtype=a.dtype, device=a.device)

    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)

    rank = a_exp.dim()
    shape_t, a_strides_t, b_strides_t = _make_index_tensors(out_shape, a_exp.stride(), b_exp.stride(), a.device)

    # DTYPE_KIND: 0 float, 1 int, 2 bool
    if a.dtype.is_floating_point:
        dtype_kind = 0
    elif a.dtype == torch.bool:
        dtype_kind = 2
    else:
        dtype_kind = 1

    # Launch configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _minimum_broadcast_kernel[grid](
        a_exp, b_exp, out,
        n_elements,
        shape_t,
        a_strides_t, b_strides_t,
        RANK=rank,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE_KIND=dtype_kind,
    )

    return out