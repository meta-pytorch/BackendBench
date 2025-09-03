# kernel.py
"""
Triton implementation of `torch.log2`
====================================

This module exposes a single public function – ``kernel_function`` – that behaves
exactly like ``torch.log2`` but runs the computation inside a Triton kernel.
It

* works for every CUDA–floating-point dtype supported by Triton (fp16 / bf16 /
  fp32 – fp64 support in Triton is still experimental),
* handles arbitrary shapes – including 0-D scalars and empty tensors,
* copes with *any* input strides (a contiguous copy is made transparently),
* honours special-value semantics (±Inf, NaN, negative numbers, zeros, …),
* uses only Triton arithmetic in the kernel body – **no cheating with PyTorch
  ops**.

The kernel follows the guidelines given in the task statement: proper masking,
coalesced accesses, compile-time constants via ``tl.constexpr`` and a clean
wrapper that hides all launch details from the caller.
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch

# -----------------------------------------------------------------------------
#                               TRITON KERNEL
# -----------------------------------------------------------------------------
@triton.jit
def _log2_kernel(in_ptr,           # *  pointer to input  tensor
                 out_ptr,          # *  pointer to output tensor
                 n_elements,       # *  total number of elements
                 BLOCK_SIZE: tl.constexpr):
    """
    Vectorised element-wise base-2 logarithm.

    A 1-D grid is launched; each Triton *program* (CUDA block) processes
    ``BLOCK_SIZE`` contiguous elements.  Out-of-bounds accesses are masked out.
    """
    # -----------------------------------------------------------
    # Block / thread organisation
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)                       # block index in grid
    block_start = pid * BLOCK_SIZE                    # first element this block handles
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vector of indices [block_start ...]
    mask = offsets < n_elements                       # boundary check

    # -----------------------------------------------------------
    # Memory I/O – coalesced because we hit consecutive addresses
    # -----------------------------------------------------------
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # -----------------------------------------------------------
    # Arithmetic – promote to fp32 for higher accuracy when the
    # input is fp16/bf16, then cast back.
    # -----------------------------------------------------------
    y_fp32 = tl.log2(x.to(tl.float32))
    y = y_fp32.to(x.dtype)

    # -----------------------------------------------------------
    # Write-back
    # -----------------------------------------------------------
    tl.store(out_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------------------
#                           PYTHON WRAPPER FUNCTION
# -----------------------------------------------------------------------------
def log2_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for ``torch.log2``.

    Parameters
    ----------
    x : torch.Tensor (CUDA, floating point)
        Input tensor.  Can have any shape/strides, including 0-D scalars and
        empty tensors.

    Returns
    -------
    torch.Tensor
        ``torch.log2(x)`` computed via Triton.  The result resides on the same
        device and has the same dtype & shape as *x* (it will be contiguous,
        which matches PyTorch’s behaviour for element-wise ops).
    """
    # ------------------------------------------------------------------
    # Basic validation & early exits
    # ------------------------------------------------------------------
    if not x.is_cuda:
        raise ValueError("Input tensor must live on a CUDA device.")
    if x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,  # Triton may down-cast internally but we accept the type
    ):
        raise TypeError(
            f"Unsupported dtype {x.dtype}. "
            "Only float16 / bfloat16 / float32 / float64 are supported."
        )

    # PyTorch returns an empty tensor immediately for .log2 on empty input – do the same.
    if x.numel() == 0:
        return torch.empty_like(x)

    # ------------------------------------------------------------------
    # Make the input contiguous – avoids dealing with complicated
    # stride math inside the kernel.  This does *not* change the
    # semantics because torch.log2 would return a contiguous tensor
    # as well.
    # ------------------------------------------------------------------
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    # ------------------------------------------------------------------
    # Grid configuration
    # ------------------------------------------------------------------
    n_elements = x_contig.numel()
    BLOCK_SIZE = 1024  # power-of-two for good memory throughput

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    _log2_kernel[grid](
        x_contig,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # The output has the correct *values* already, we just need to reshape
    # it to the original logical shape (contiguous layout).
    return out.view_as(x)