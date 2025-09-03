# kernel.py
"""
Triton implementation of the in-place SiLU (Sigmoid-Linear-Unit) activation

      y = x * sigmoid(x)     with   sigmoid(x) = 1 / (1 + exp(-x))

The public entry-point `kernel_function` behaves exactly like
`torch.ops.aten.silu_.default`, i.e. it mutates its input tensor **in-place**
and returns that very same tensor.

Key features
------------
• Works for every rank / stride pattern that PyTorch supports (positive strides)
• No use of any PyTorch math in the kernel – everything is computed with
  Triton primitives (`tl.load`, `tl.store`, `tl.exp`, …)
• Handles all boundary conditions via masking
• Coalesced accesses for contiguous inputs; still correct for strided ones
• Written following the “Triton Kernel Programming Guidelines” supplied
"""

from __future__ import annotations

import math
from typing import List

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------#
# Kernel – runs on the GPU
# -----------------------------------------------------------------------------#

MAX_DIMS = 8          # we support up to 8-D tensors
BLOCK_SIZE = 1024     # elements handled by one Triton *program*  (power of 2)


@triton.jit
def _silu_kernel(
    ptr,                         # *T*        – pointer to tensor data
    n_elements: tl.int32,        # total number of elements in tensor

    # --- shape[d] -------------------------------------------------------------#
    S0: tl.int32, S1: tl.int32, S2: tl.int32, S3: tl.int32,
    S4: tl.int32, S5: tl.int32, S6: tl.int32, S7: tl.int32,

    # --- stride[d] (in *elements*,  not bytes) --------------------------------#
    STR0: tl.int32, STR1: tl.int32, STR2: tl.int32, STR3: tl.int32,
    STR4: tl.int32, STR5: tl.int32, STR6: tl.int32, STR7: tl.int32,

    # --- row-major contiguous strides used to decode a linear index -----------#
    RS0: tl.int32, RS1: tl.int32, RS2: tl.int32, RS3: tl.int32,
    RS4: tl.int32, RS5: tl.int32, RS6: tl.int32, RS7: tl.int32,

    BLOCK: tl.constexpr                       # block size (compile-time const)
):
    """Vectorised in-place SiLU.

    The kernel linearly enumerates all `n_elements` indices, then maps each
    linear index to the corresponding *multi-dimensional* index using
    user-provided shapes/strides.  This allows us to deal with arbitrary
    (non-contiguous) tensors without additional gather/scatter indirection.
    """
    # --------------------- compute global indices ---------------------------- #
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offs = block_start + tl.arange(0, BLOCK)              # [BLOCK] int32
    mask = offs < n_elements

    # We will successively peel off digits of the linear index to obtain the
    # coordinate for each dimension d and accumulate the element offset using
    # the *real* (possibly non-contiguous) PyTorch stride.
    #
    # offset_in_elements = ∑   idx_d * stride_d
    #
    # where   idx_d = (remaining // row_stride_d)
    #         remaining %= row_stride_d
    #
    # NOTE:  row_stride_d  is   ∏_{k > d}  shape[k]
    remaining = offs
    offset_elems = tl.zeros_like(offs)        # running element offset

    # --- dim 0 ----------------------------------------------------------------
    idx = remaining // RS0
    remaining -= idx * RS0
    offset_elems += idx * STR0

    # --- dim 1 ----------------------------------------------------------------
    idx = remaining // RS1
    remaining -= idx * RS1
    offset_elems += idx * STR1

    # --- dim 2 ----------------------------------------------------------------
    idx = remaining // RS2
    remaining -= idx * RS2
    offset_elems += idx * STR2

    # --- dim 3 ----------------------------------------------------------------
    idx = remaining // RS3
    remaining -= idx * RS3
    offset_elems += idx * STR3

    # --- dim 4 ----------------------------------------------------------------
    idx = remaining // RS4
    remaining -= idx * RS4
    offset_elems += idx * STR4

    # --- dim 5 ----------------------------------------------------------------
    idx = remaining // RS5
    remaining -= idx * RS5
    offset_elems += idx * STR5

    # --- dim 6 ----------------------------------------------------------------
    idx = remaining // RS6
    remaining -= idx * RS6
    offset_elems += idx * STR6

    # --- dim 7 ----------------------------------------------------------------
    # RS7 == 1 by construction – no modulo needed afterwards
    idx = remaining // RS7
    offset_elems += idx * STR7

    # ----------------------- load -> compute -> store ------------------------ #
    ptrs = ptr + offset_elems                  # true memory addresses
    x = tl.load(ptrs, mask=mask)

    # Promote to f32 for better numeric stability,  then down-cast again.
    x_f32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_f32))
    y_f32 = x_f32 * sig
    y = y_f32.to(x.dtype)

    tl.store(ptrs, y, mask=mask)


# -----------------------------------------------------------------------------#
# Public wrapper – runs on the host (Python)
# -----------------------------------------------------------------------------#
def _build_row_major_contiguous_strides(shape: List[int]) -> List[int]:
    """
    For a given `shape` return the row-major contiguous strides
       RS[d] = ∏_{k>d} shape[k]
    Needed to decode a flat linear index inside the kernel.
    """
    rs = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        rs[d] = rs[d + 1] * shape[d + 1]
    return rs


def _pad_to_max_dims(lst: List[int], pad_value: int, *, max_len: int = MAX_DIMS) -> List[int]:
    """Right-pad `lst` with `pad_value` until its length is `max_len`."""
    return lst + [pad_value] * (max_len - len(lst))


def silu__kernel_impl(x: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """
    Apply SiLU to `x` *in-place* using a Triton kernel.

    Parameters
    ----------
    x : torch.Tensor
        CUDA tensor to be modified in-place

    Returns
    -------
    torch.Tensor
        The same tensor object (`x`) after mutation
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must live on a CUDA device.")
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError("Only float16 / float32 / bfloat16 tensors are supported.")

    # Early exit for empty tensors
    n_elements = x.numel()
    if n_elements == 0:
        return x

    # Collect (and pad) shape + stride information
    shape: List[int] = list(x.shape)
    stride: List[int] = list(x.stride())           # element-strides (not bytes)

    if len(shape) > MAX_DIMS:
        raise NotImplementedError(
            f"Tensor rank {len(shape)} exceeds hard limit of {MAX_DIMS}"
        )

    shape = _pad_to_max_dims(shape, 1)             # 1   keeps formula valid
    stride = _pad_to_max_dims(stride, 0)           # 0   -> contributes nothing

    row_major = _build_row_major_contiguous_strides(shape)

    # Kernel launch
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # 1-D launch

    _silu_kernel[grid](
        x,                              # ptr
        n_elements,                     # total elements

        # --- shapes ----------------------------------------------------------#
        *shape,                         # S0 … S7

        # --- (PyTorch) strides ----------------------------------------------#
        *stride,                        # STR0 … STR7

        # --- row-major contiguous strides -----------------------------------#
        *row_major,                     # RS0 … RS7

        BLOCK=BLOCK_SIZE,               # constexpr
    )

    return x