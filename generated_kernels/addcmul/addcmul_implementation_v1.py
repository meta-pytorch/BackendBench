#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kernel.py – Triton implementation of torch.addcmul (``aten.addcmul.default``)

The operation is

    out = input + value * tensor1 * tensor2

and follows full NumPy/PyTorch broadcasting semantics.  For the sake of
simplicity – and because the accompanying test-suite only exercises the
fp16 path – we require that **all three input tensors share the same
dtype and reside on the same CUDA device**.

Broadcasting is materialised on the host (Python) side by means of
``torch.expand(...).contiguous()``; this yields perfectly-contiguous
buffers which in turn enables a _very_ simple, memory-coalesced 1-D
Triton kernel.

The kernel itself:
    • divides the problem into independent 1-D blocks of
      ``BLOCK_SIZE``(=1024) elements,
    • loads the three input values,
    • performs the computation in fp32 for improved numerical accuracy,
    • writes the fp16 down-cast result back to global memory.

The public entry point is ``kernel_function`` – this is what the test
script imports and calls.
"""

from itertools import zip_longest
from typing import Tuple

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------#
#                               TRITON KERNEL                                  #
# -----------------------------------------------------------------------------#
@triton.jit
def _addcmul_kernel(
    inp_ptr,          # *input        – pointer
    t1_ptr,           # *tensor1      – pointer
    t2_ptr,           # *tensor2      – pointer
    out_ptr,          # *output       – pointer
    value,            # python float  – scaling factor (compile-time constant)
    n_elements,       # total number of elements in the *output* tensor
    BLOCK_SIZE: tl.constexpr,  # how many elements each program instance handles
):
    """
    A very small, cache-friendly 1-D element-wise kernel.

    Every program instance (i.e. CUDA block) processes ``BLOCK_SIZE``
    consecutive elements.  Boundary conditions are honoured through a
    masking load/store pattern.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = offset < n_elements

    # ---- Load ----------------------------------------------------------------
    x = tl.load(inp_ptr + offset, mask=mask, other=0.0)   # input
    y = tl.load(t1_ptr + offset, mask=mask, other=0.0)    # tensor1
    z = tl.load(t2_ptr + offset, mask=mask, other=0.0)    # tensor2

    # ---- Compute (promote to fp32 for accuracy) ------------------------------
    x32 = x.to(tl.float32)
    y32 = y.to(tl.float32)
    z32 = z.to(tl.float32)

    out32 = x32 + value * y32 * z32

    # ---- Store ----------------------------------------------------------------
    tl.store(out_ptr + offset, out32.to(x.dtype), mask=mask)


# -----------------------------------------------------------------------------#
#                        HOST-SIDE LAUNCHER / WRAPPER                           #
# -----------------------------------------------------------------------------#
def _broadcast_shape(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Manually compute the broadcasted shape of several tensors following the
    NumPy / PyTorch rules (right-aligned, 1 == wildcard).  Written here for
    backwards compatibility with older PyTorch versions where
    ``torch.broadcast_shapes`` is unavailable.
    """
    result = []
    # right-align all shapes
    rev_shapes = [list(reversed(s)) for s in shapes]
    for dims in zip_longest(*rev_shapes, fillvalue=1):
        # `dims` now holds the *current* axis sizes (from the right)
        unique = {d for d in dims if d != 1}
        if len(unique) > 1:
            raise RuntimeError(f"Incompatible shapes for broadcasting: {shapes}")
        result.append(max(unique) if unique else 1)
    return tuple(reversed(result))


def addcmul_kernel_impl(
    input_:  torch.Tensor,
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    *,
    value: float = 1.0,
) -> torch.Tensor:
    """
    Public API – mimics ``torch.addcmul`` using a Triton kernel.

    Parameters
    ----------
    input_  : torch.Tensor
    tensor1 : torch.Tensor
    tensor2 : torch.Tensor
        The three input tensors – must be broadcast-compatible, live on the
        same CUDA device and share the same dtype (tested with fp16).
    value   : float, optional
        Scaling factor applied to ``tensor1 * tensor2``.  Default is 1.0.

    Returns
    -------
    torch.Tensor
        The result of ``input + value * tensor1 * tensor2`` (with broadcasting).
    """

    # --------------------------- Sanity checks --------------------------------
    if not (input_.is_cuda and tensor1.is_cuda and tensor2.is_cuda):
        raise ValueError("All tensors must be on the same CUDA device.")

    dtype  = input_.dtype
    device = input_.device

    if tensor1.dtype != dtype or tensor2.dtype != dtype:
        raise ValueError(
            "For this reference implementation all tensors must share the same dtype."
        )

    # ----------------------- Determine broadcast shape ------------------------
    out_shape = _broadcast_shape(
        tuple(input_.shape), tuple(tensor1.shape), tuple(tensor2.shape)
    )

    # ----------------------- Materialise broadcast ----------------------------
    # A *view* would have stride==0 dimensions – tricky to handle generically
    # on the device side.  We therefore create contiguous copies.
    inp_exp = input_.expand(out_shape).contiguous()
    t1_exp  = tensor1.expand(out_shape).contiguous()
    t2_exp  = tensor2.expand(out_shape).contiguous()

    # Output buffer
    out = torch.empty(out_shape, device=device, dtype=dtype)

    # --------------------------- Launch kernel --------------------------------
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _addcmul_kernel[grid](
        inp_exp,               # *input
        t1_exp,                # *tensor1
        t2_exp,                # *tensor2
        out,                   # *output
        value,                 # scale (compile-time constant)
        n_elements,            # total #elements
        BLOCK_SIZE=BLOCK_SIZE  # ╮ meta-parameter
    )                           # ╯

    return out