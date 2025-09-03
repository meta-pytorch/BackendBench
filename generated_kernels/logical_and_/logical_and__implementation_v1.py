# kernel.py
# Triton in-place logical_and_ kernel with broadcasting and arbitrary strides.
# Follows Triton programming guidelines and passes the provided tests.

import torch
import triton
import triton.language as tl


# We support up to MAX_DIMS tensor dimensions by right-aligning shapes/strides and padding leading dims.
MAX_DIMS = 8


@triton.jit
def _logical_and_inplace_kernel(
    lhs_ptr,                # *bool
    rhs_ptr,                # *bool
    shape_ptr,              # *int64, length MAX_DIMS (right-aligned, padded with 1s)
    lhs_strides_ptr,        # *int64, length MAX_DIMS (right-aligned, padded)
    rhs_strides_ptr,        # *int64, length MAX_DIMS (right-aligned, padded; broadcast dims have stride=0)
    n_elements,             # int64
    BLOCK_SIZE: tl.constexpr,
    MAXR: tl.constexpr,     # number of dims (compile-time), equals MAX_DIMS from host
):
    # 1D launch: each program handles a block of BLOCK_SIZE elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create vector of indices for this block
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Use int64 for address arithmetic
    offs = offs.to(tl.int64)

    # Decode linear indices into multi-dimensional indices using mixed radix (right-aligned dims)
    # and compute the source (rhs) and destination (lhs) memory offsets based on strides.
    rem = offs
    off_lhs = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    off_rhs = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Loop over dimensions from last to first (right-aligned)
    # shape_ptr[i] is the size of dimension i (i in [0..MAXR-1]); leading dims can be 1.
    for d in range(MAXR - 1, -1, -1):
        dim_size = tl.load(shape_ptr + d)  # int64 scalar
        # Avoid division/modulo by zero: if n_elements > 0, every dim_size must be >= 1.
        # The host guarantees no launch when n_elements == 0.
        idx_d = rem % dim_size
        rem = rem // dim_size

        stride_l = tl.load(lhs_strides_ptr + d)
        stride_r = tl.load(rhs_strides_ptr + d)

        off_lhs += idx_d * stride_l
        off_rhs += idx_d * stride_r

    # Compute final pointers
    lhs_addrs = lhs_ptr + off_lhs
    rhs_addrs = rhs_ptr + off_rhs

    # Load, compute boolean AND, and store back in-place
    a = tl.load(lhs_addrs, mask=mask, other=0)  # bool/int1
    b = tl.load(rhs_addrs, mask=mask, other=0)  # bool/int1
    res = a & b
    tl.store(lhs_addrs, res, mask=mask)


def _right_align_and_pad(shape, strides, target_ndims):
    """
    Right-align shapes/strides to target_ndims:
      - For missing leading dims, shape=1 and stride arbitrary; we set to 0 for safety.
      - Return lists of length target_ndims.
    """
    nd = len(shape)
    pad = target_ndims - nd
    # Stride for a size-1 dim can be anything; we set 0 for simplicity.
    shape_padded = [1] * pad + list(shape)
    strides_padded = [0] * pad + list(strides)
    return shape_padded, strides_padded


def _make_broadcasted_rhs_strides(lhs_shape_ra, rhs_shape_ra, rhs_strides_ra):
    """
    Create broadcasted RHS strides:
      - If rhs dimension is 1 (broadcasted), use stride 0
      - Else, use the given stride
    Assumes right-aligned arrays (same length).
    """
    out = []
    for ls, rs, st in zip(lhs_shape_ra, rhs_shape_ra, rhs_strides_ra):
        if rs == 1 and ls != 1:
            out.append(0)
        else:
            out.append(st)
    return out


def logical_and__kernel_impl(lhs: torch.Tensor, rhs: torch.Tensor):
    """
    In-place boolean logical AND on GPU using Triton with broadcasting and arbitrary strides.

    Args:
        lhs: torch.bool CUDA tensor to be updated in-place.
        rhs: torch.bool CUDA tensor broadcastable to lhs.shape.

    Returns:
        The same tensor as lhs (mutated in-place).
    """
    if not lhs.is_cuda or not rhs.is_cuda:
        raise ValueError("Both tensors must be CUDA tensors.")
    if lhs.dtype != torch.bool or rhs.dtype != torch.bool:
        raise TypeError("Both tensors must have dtype=torch.bool.")
    # Broadcastability check (PyTorch-like)
    lhs_shape = list(lhs.shape)
    rhs_shape = list(rhs.shape)
    nd = max(len(lhs_shape), len(rhs_shape))
    lhs_shape_ra, lhs_strides_ra = _right_align_and_pad(lhs_shape, lhs.stride(), nd)
    rhs_shape_ra, rhs_strides_ra = _right_align_and_pad(rhs_shape, rhs.stride(), nd)

    # Validate broadcastability
    for ls, rs in zip(lhs_shape_ra, rhs_shape_ra):
        if not (rs == 1 or rs == ls):
            raise ValueError(f"rhs shape {tuple(rhs.shape)} is not broadcastable to lhs shape {tuple(lhs.shape)}")

    # Create broadcasted rhs strides (stride 0 for broadcasted dims)
    rhs_strides_brd = _make_broadcasted_rhs_strides(lhs_shape_ra, rhs_shape_ra, rhs_strides_ra)

    # Number of elements
    n_elements = lhs.numel()
    if n_elements == 0:
        # Nothing to do; return lhs to preserve aliasing semantics
        return lhs

    # We right-align to MAX_DIMS for the kernel by padding with leading dims
    if nd > MAX_DIMS:
        # Optional: support more dims by increasing MAX_DIMS if needed
        raise ValueError(f"Exceeded MAX_DIMS={MAX_DIMS}; got {nd} dims")

    pad = MAX_DIMS - nd
    shape_for_kernel = [1] * pad + lhs_shape_ra
    lhs_strides_for_kernel = [0] * pad + [int(s) for s in lhs_strides_ra]
    rhs_strides_for_kernel = [0] * pad + [int(s) for s in rhs_strides_brd]

    device = lhs.device
    # Device arrays for shapes/strides (int64)
    shape_dev = torch.tensor(shape_for_kernel, dtype=torch.int64, device=device)
    lhs_strides_dev = torch.tensor(lhs_strides_for_kernel, dtype=torch.int64, device=device)
    rhs_strides_dev = torch.tensor(rhs_strides_for_kernel, dtype=torch.int64, device=device)

    # Launch configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch Triton kernel
    _logical_and_inplace_kernel[grid](
        lhs, rhs,
        shape_dev, lhs_strides_dev, rhs_strides_dev,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        MAXR=MAX_DIMS,
        # Optional tuning knobs:
        num_warps=4,
        num_stages=2,
    )

    # Return the mutated LHS tensor (preserving aliasing semantics)
    return lhs