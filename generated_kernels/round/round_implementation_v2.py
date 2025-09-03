# ---------------------------------------------------------------------------------------
#  kernel.py
#
#  Triton implementation of torch.round / aten.round.default
#  ---------------------------------------------------------
#  * Rounds every element to the nearest integer (ties-to-even a.k.a “banker’s” rounding)
#  * Supports float16 / bfloat16 / float32 tensors of any shape
#  * Works for 0-D scalars, contiguous & non-contiguous tensors
#  * The heavy-lifting is done inside a Triton kernel that only uses tl.* ops
#  * A python wrapper `kernel_function` takes care of bookkeeping / launch
# ---------------------------------------------------------------------------------------
"""
Round-to-nearest-even (banker’s rounding) implemented with Triton.

Usage
-----
>>> import torch, kernel                                # noqa: E402
>>> x = torch.randn(1024, device='cuda', dtype=torch.bfloat16) * 23.7
>>> y = kernel.kernel_function(x)                       # identical to torch.round(x)
>>> torch.allclose(y, torch.round(x))
True
"""
from __future__ import annotations

import triton
import triton.language as tl
import torch


# ---------------------------------------------------------------------------------------
#  Triton kernel
# ---------------------------------------------------------------------------------------
@triton.jit
def _round_kernel(
    in_ptr,                                            # (*) pointer to input  tensor
    out_ptr,                                           # (*) pointer to output tensor
    n_elements,                                        # total number of elements
    BLOCK_SIZE: tl.constexpr,                          # how many elements each block processes
):
    """
    Element-wise round-to-nearest-even (banker’s rounding).

    The algorithm is implemented in float32 for numerical robustness and then cast
    back to the original dtype before writing results.
    """
    pid = tl.program_id(axis=0)                       # 1-D grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # shape [BLOCK_SIZE]
    mask = offsets < n_elements                       # guard against out-of-bounds

    # ------------------------------------------------------------------
    #  Load ----------------------------------------------------------------
    # ------------------------------------------------------------------
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # ------------------------------------------------------------------
    #  Compute (float32 math) -------------------------------------------
    #    Algorithm:
    #       f  = floor(x)
    #       frac = x - f
    #       if frac > 0.5                        →  f + 1
    #       if frac < 0.5                        →  f
    #       if frac == 0.5                       →  f + (f is odd)     (ties-to-even)
    # ------------------------------------------------------------------
    x_f32 = x.to(tl.float32)

    f = tl.math.floor(x_f32)
    frac = x_f32 - f
    half = 0.5

    gt_half = frac > half                           #  frac  > 0.5 ?
    eq_half = frac == half                          #  frac == 0.5 ?

    # `f` is an integer value in float32.  Convert to int32 to test parity.
    f_int = f.to(tl.int32)
    is_odd = (f_int & 1) != 0                       # True  ↔  odd

    inc_from_tie = eq_half & is_odd                 # need +1 because tie & odd
    inc_general  = gt_half                          # need +1 because frac > 0.5
    need_inc = inc_general | inc_from_tie           # logical “or” (bool tensor)

    rounded_f32 = f + need_inc.to(tl.float32)
    rounded = rounded_f32.to(x.dtype)               # cast back to original dtype

    # ------------------------------------------------------------------
    #  Store --------------------------------------------------------------
    # ------------------------------------------------------------------
    tl.store(out_ptr + offsets, rounded, mask=mask)


# ---------------------------------------------------------------------------------------
#  Public wrapper
# ---------------------------------------------------------------------------------------
def round_kernel_impl(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    A drop-in replacement for `torch.round` implemented with Triton.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor to round.  Must reside on a CUDA device and have dtype
        float16, bfloat16 or float32.

    Returns
    -------
    torch.Tensor
        A tensor containing the rounded values.  Strides / memory-format of
        the input are preserved.
    """
    if not input_tensor.is_cuda:
        raise ValueError("kernel_function only works on CUDA tensors.")
    if input_tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {input_tensor.dtype}. "
            "Supported dtypes: float16, bfloat16, float32."
        )

    # We compute on a *contiguous* copy for simpler indexing.
    inp_contig = input_tensor.contiguous()
    out_contig = torch.empty_like(inp_contig)

    # Launch parameters ---------------------------------------------------
    n_elems = inp_contig.numel()
    BLOCK_SIZE = 1024                               # good default, power-of-2
    grid = (triton.cdiv(n_elems, BLOCK_SIZE),)      # 1-D launch

    _round_kernel[grid](
        inp_contig, out_contig,                     # pointers
        n_elems,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,                                # reasonable default for 1-D kernels
    )

    # If the original tensor is contiguous we can return the contiguous output directly.
    if input_tensor.is_contiguous():
        return out_contig

    # Otherwise, allocate a tensor with *identical* shape & strides and copy.
    out_strided = torch.empty_strided(
        size=input_tensor.shape,
        stride=input_tensor.stride(),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    out_strided.copy_(out_contig)
    return out_strided