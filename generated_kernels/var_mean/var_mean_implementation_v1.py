# kernel.py
import math
from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl


"""
Var/Mean kernel (aten.var_mean.correction) in Triton

This file implements a Triton kernel that computes, for an input tensor:
- variance with arbitrary Bessel correction (int "correction")
- mean
over one or more reduction dimensions (dim=None or list of ints), with keepdim
behavior matching PyTorch:
    torch.ops.aten.var_mean.correction(x, dim, correction=..., keepdim=...)

Key properties:
- Works for contiguous and non-contiguous inputs (uses strides)
- Supports multiple reduction dimensions and dim=None (full reduction)
- Handles negative dims and keepdim True/False
- Numerically: computes sum and sum of squares in float32, then derives mean
  and (corrected) variance at the end. Results are cast back to input dtype.
- When (N - correction) == 0, variance becomes NaN (as in PyTorch).

Implementation notes:
- One Triton program computes var/mean for one kept-dim output coordinate.
- Reduction across all reduced dims is flattened to a single [0..NRED) loop,
  iterated in tiles of BLOCK_R.
- For address computation, we use mixed-radix decomposition with the sizes and
  strides of the reduction dims passed as arrays.
- We precompute "base_offsets" on the host for each output element to avoid
  reconstructing the kept-dim indices inside the kernel.
"""


@triton.jit
def _varmean_kernel(
    x_ptr,                              # * Input tensor pointer
    var_ptr,                            # * Output variance pointer (same dtype as input)
    mean_ptr,                           # * Output mean pointer (same dtype as input)
    base_offsets_ptr,                   # * int64 base offsets for each output element (length = NUM_OUT)
    red_shapes_ptr,                     # * int32 reduction sizes (length = MAX_RED_DIMS)
    red_strides_ptr,                    # * int64 reduction strides (length = MAX_RED_DIMS)
    NUM_OUT,                            # number of output elements (int32)
    NRED,                               # product(reduction sizes) (int32)
    correction_f32,                     # float32 correction (Bessel correction)
    BLOCK_R: tl.constexpr,              # tile size along reduction
    MAX_RED_DIMS: tl.constexpr,         # compile-time max number of reduction dims
):
    """
    One program computes var/mean for one kept-dim coordinate (one output element).
    It reduces over all reduction dims (flattened to [0..NRED)).
    """
    pid = tl.program_id(axis=0)
    if pid >= NUM_OUT:
        return

    # Load base input offset for this output element (in elements)
    base_off = tl.load(base_offsets_ptr + pid, mask=True).to(tl.int64)

    # Accumulators in float32 for numerical stability
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)
    acc_count = tl.zeros((), dtype=tl.float32)

    # Iterate over the reduction space in tiles of BLOCK_R
    for r_start in tl.range(0, NRED, BLOCK_R):
        offs = r_start + tl.arange(0, BLOCK_R)  # int32 vector [BLOCK_R]
        mask = offs < NRED

        # Build offsets inside the reduction space via mixed-radix decomposition
        tmp = offs.to(tl.int64)
        red_offs = tl.zeros([BLOCK_R], dtype=tl.int64)
        # Loop over MAX_RED_DIMS (padded with size=1, stride=0)
        for i in range(MAX_RED_DIMS):
            size_i = tl.load(red_shapes_ptr + i).to(tl.int64)
            stride_i = tl.load(red_strides_ptr + i)  # already int64
            coor_i = tmp % size_i
            tmp = tmp // size_i
            red_offs += coor_i * stride_i

        # Gather input values for this tile
        ptrs = x_ptr + (base_off + red_offs)
        x_vals = tl.load(ptrs, mask=mask, other=0)

        # Accumulate in float32
        x_f32 = x_vals.to(tl.float32)
        sum_tile = tl.sum(x_f32, axis=0)
        sumsq_tile = tl.sum(x_f32 * x_f32, axis=0)
        # Count valid lanes in this tile
        cnt_tile = tl.sum(tl.where(mask, 1.0, 0.0), axis=0).to(tl.float32)

        acc_sum += sum_tile
        acc_sumsq += sumsq_tile
        acc_count += cnt_tile

    # Final mean and variance (with correction)
    mean_f32 = acc_sum / acc_count
    m2 = acc_sumsq - (acc_sum * acc_sum) / acc_count
    denom = acc_count - correction_f32
    var_f32 = m2 / denom

    # Cast back to output dtype based on pointer element type
    out_dtype = mean_ptr.dtype.element_ty
    mean_out = mean_f32.to(out_dtype)
    var_out = var_f32.to(out_dtype)

    tl.store(mean_ptr + pid, mean_out, mask=True)
    tl.store(var_ptr + pid, var_out, mask=True)


def _normalize_dims(dim: Optional[Union[int, List[int]]], ndim: int) -> List[int]:
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        dim = [dim]
    # normalize negatives and deduplicate while preserving order
    seen = set()
    norm = []
    for d in dim:
        if d < 0:
            d += ndim
        if d < 0 or d >= ndim:
            raise IndexError(f"Dimension out of range: {d} for ndim={ndim}")
        if d not in seen:
            seen.add(d)
            norm.append(d)
    return norm


def _compute_base_offsets(shape: List[int], strides: List[int], keep_dims: List[int]) -> Tuple[torch.Tensor, int]:
    """
    Compute base input offsets (in elements) for every output element corresponding
    to the kept dimensions. This is done on CPU with simple integer arithmetic and
    returned as a CUDA int64 tensor.

    Args:
        shape: full input shape (list of ints)
        strides: full input strides in elements (list of ints)
        keep_dims: list of axes to keep (complement of reduction axes)

    Returns:
        (base_offsets_cuda, num_out)
    """
    if len(keep_dims) == 0:
        base_offsets = torch.tensor([0], dtype=torch.int64)
        return base_offsets, 1

    keep_sizes = [int(shape[d]) for d in keep_dims]
    keep_strides = [int(strides[d]) for d in keep_dims]
    num_out = 1
    for s in keep_sizes:
        num_out *= s

    base_offsets = torch.empty(num_out, dtype=torch.int64)
    # Map linear index -> multi-index (row-major, last dimension fastest)
    for linear in range(num_out):
        rest = linear
        off = 0
        # Decompose from last kept dim to first
        for i in range(len(keep_dims) - 1, -1, -1):
            size_i = keep_sizes[i]
            if size_i > 0:
                idx_i = rest % size_i
                rest //= size_i
            else:
                idx_i = 0
            off += idx_i * keep_strides[i]
        base_offsets[linear] = off

    return base_offsets, num_out


def _prepare_reduction_meta(
    shape: List[int],
    strides: List[int],
    red_dims: List[int],
    max_red_dims: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Prepare reduction shapes and strides arrays, padded to max_red_dims.
    Returns CUDA tensors:
      - red_shapes (int32, length=max_red_dims)
      - red_strides (int64, length=max_red_dims)
      - NRED (product of reduction sizes)
    """
    red_shapes_list = [int(shape[d]) for d in red_dims]
    red_strides_list = [int(strides[d]) for d in red_dims]
    NRED = 1
    for s in red_shapes_list:
        NRED *= s

    # Pad to max_red_dims with size=1 (neutral for mixed-radix) and stride=0
    while len(red_shapes_list) < max_red_dims:
        red_shapes_list.append(1)
        red_strides_list.append(0)

    red_shapes = torch.tensor(red_shapes_list, dtype=torch.int32, device=device)
    red_strides = torch.tensor(red_strides_list, dtype=torch.int64, device=device)
    return red_shapes, red_strides, int(NRED)


def var_mean_kernel_impl(
    x: torch.Tensor,
    dim: Optional[Union[int, List[int]]] = None,
    *,
    correction: int = 0,
    keepdim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function that launches the Triton var_mean kernel.

    Args:
        x: Input tensor (CUDA). Tested with bfloat16 but works with other floating dtypes.
        dim: None or list of ints specifying reduction dimensions.
        correction: Bessel correction (e.g., 0 for biased, 1 for unbiased).
        keepdim: Whether to keep reduced dimensions with size 1.

    Returns:
        (var, mean): Tensors with shapes/dtypes equivalent to
                     torch.ops.aten.var_mean.correction(x, dim, correction, keepdim).
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    device = x.device

    # Handle empty tensors by delegating shape/dtype behavior to PyTorch
    if x.numel() == 0:
        ref_var, ref_mean = torch.ops.aten.var_mean.correction(x, dim, correction=correction, keepdim=keepdim)
        return ref_var, ref_mean

    ndim = x.dim()
    shape = list(x.shape)
    strides = list(x.stride())  # in elements

    red_dims = _normalize_dims(dim, ndim)
    keep_dims = [d for d in range(ndim) if d not in red_dims]

    # Compute output shapes
    if keepdim:
        out_shape = [1 if i in red_dims else shape[i] for i in range(ndim)]
    else:
        out_shape = [shape[i] for i in keep_dims]

    # Allocate outputs (contiguous) with same dtype as input
    var_out = torch.empty(out_shape, dtype=x.dtype, device=device)
    mean_out = torch.empty(out_shape, dtype=x.dtype, device=device)

    # Precompute base offsets for kept dims
    base_offsets_cpu, num_out = _compute_base_offsets(shape, strides, keep_dims)
    base_offsets = base_offsets_cpu.to(device=device, non_blocking=True)

    # Prepare reduction metadata
    MAX_RED_DIMS = 8  # compile-time constant upper bound
    red_shapes, red_strides, NRED = _prepare_reduction_meta(shape, strides, red_dims, MAX_RED_DIMS, device=device)

    # Special case: no reduction (dim=[]), return elementwise var=0, mean=x
    if len(red_dims) == 0:
        mean_out.copy_(x)
        var_out.zero_()
        return var_out, mean_out

    # Grid: one program per output element
    grid = (num_out,)

    # Choose a reasonable BLOCK_R (power of two)
    if NRED >= 1024:
        BLOCK_R = 1024
    elif NRED >= 512:
        BLOCK_R = 512
    elif NRED >= 256:
        BLOCK_R = 256
    else:
        BLOCK_R = 128

    _varmean_kernel[grid](
        x,
        var_out,
        mean_out,
        base_offsets,
        red_shapes,
        red_strides,
        num_out,                 # NUM_OUT
        NRED,                    # NRED
        float(correction),       # correction_f32
        BLOCK_R=BLOCK_R,
        MAX_RED_DIMS=MAX_RED_DIMS,
    )

    return var_out, mean_out