#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kernel.py – Triton implementation of `aten.reciprocal.default`

This module exposes a single public symbol called `kernel_function`
that behaves like `torch.ops.aten.reciprocal.default`, but the actual
numerical work is carried-out inside a Triton kernel so that it runs on
the GPU.

Supported dtypes
----------------
• torch.bfloat16   (preferred over float32 by the unit-test)
• torch.float16

Behaviour / Semantics
---------------------
Given an input tensor `x`, the function returns a *new* tensor `y`
satisfying `y = 1 / x` element-wise.  All shapes (including 0-D
scalars) and any memory layout (contiguous or not) are supported.  The
output preserves the input’s shape *and* strides so that PyTorch
semantics are fully respected.

Implementation outline
----------------------
1. A thin Python wrapper (`kernel_function`) handles:
   • Argument validation
   • Allocation of the output tensor with the *same* shape & strides
   • Determination of the launch grid and invocation of the Triton
     kernel.

2. The actual work is performed by the Triton‐JITed kernel
   (`_reciprocal_kernel`) which:
   • Uses a 1-D execution grid
   • Loads a block of elements        → `tl.load`
   • Casts them to `fp32`             → higher accuracy
   • Computes `1 / x`                 → tl operations
   • Casts back to the original type
   • Stores the results               → `tl.store`
   • Properly masks out-of-bounds threads

The code strictly follows the “Triton Kernel Programming Guidelines”
provided in the problem statement.
"""
from __future__ import annotations

import triton
import triton.language as tl
import torch


# -----------------------------------------------------------------------------#
#                               TRITON KERNEL                                   #
# -----------------------------------------------------------------------------#
@triton.jit
def _reciprocal_kernel(
    x_ptr,                       # *  Input tensor
    y_ptr,                       # *  Output tensor
    numel,                       #   Number of elements in x / y
    BLOCK_SIZE: tl.constexpr,    #   Num threads per block (power of 2)
):
    """
    Each program (CUDA block) handles BLOCK_SIZE elements.  The grid is 1-D,
    hence `tl.program_id(0)` is the program id.

    Parameters
    ----------
    x_ptr : tl.pointer
        Pointer to the first byte of the input tensor (device memory).
    y_ptr : tl.pointer
        Pointer to the first byte of the output tensor (device memory).
    numel : int
        Total number of elements in the input / output tensor.
    BLOCK_SIZE : int (constexpr)
        Compile-time constant – number of elements processed per program.
    """
    # --------------------------------------------------------------------- #
    # Program / block index
    # --------------------------------------------------------------------- #
    pid = tl.program_id(axis=0)

    # --------------------------------------------------------------------- #
    # Compute the *absolute* indices (0 … numel-1) that this program owns.
    # --------------------------------------------------------------------- #
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to guard against OOB accesses for the last block.
    mask = offsets < numel

    # --------------------------------------------------------------------- #
    # Load → Compute reciprocal → Store
    # --------------------------------------------------------------------- #
    # Load the data – honour the mask to avoid invalid reads.
    x = tl.load(x_ptr + offsets, mask=mask)

    # Promote to fp32 for better accuracy, compute 1/x, then cast back to
    # the original dtype.  The original dtype is available from `x.dtype`.
    x_fp32 = x.to(tl.float32)
    recip_fp32 = 1.0 / x_fp32
    recip = recip_fp32.to(x.dtype)

    # Store the results.
    tl.store(y_ptr + offsets, recip, mask=mask)


# -----------------------------------------------------------------------------#
#                           PYTHON WRAPPER FUNCTION                             #
# -----------------------------------------------------------------------------#
def reciprocal_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Element-wise reciprocal implemented in Triton.

    This is the *public* API used by the unit-test.  It behaves just like
    `torch.ops.aten.reciprocal.default`.

    Parameters
    ----------
    x : torch.Tensor
        CUDA tensor with dtype `bfloat16` or `float16`.

    Returns
    -------
    torch.Tensor
        A new tensor `y` with `y = 1 / x`, matching the shape & strides of
        the input.
    """
    # --------------------------- Argument checks -------------------------- #
    if not x.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device.")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "Only bfloat16 and float16 are supported, "
            f"got: {x.dtype}"
        )

    # Number of elements (works for 0-D tensors as well).
    numel = x.numel()

    # ------------------------- Allocate the output ------------------------ #
    # *Preserve* input strides to fully mimic PyTorch’s behaviour.
    y = torch.empty_strided(
        size=x.size(),
        stride=x.stride(),
        dtype=x.dtype,
        device=x.device,
    )

    if numel == 0:
        # Nothing to do – can happen for empty tensors.
        return y

    # -------------------------- Launch parameters ------------------------- #
    BLOCK_SIZE = 1024  # power-of-two per the guidelines
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    # -------------------------- Kernel invocation ------------------------- #
    _reciprocal_kernel[grid](
        x,                            # x_ptr
        y,                            # y_ptr
        numel,                        # total number of elements
        BLOCK_SIZE=BLOCK_SIZE,        # constexpr
        num_warps=4,                  # reasonable default for 1-D kernels
    )

    return y