# kernel.py
"""
Triton implementation of `aten.abs.default`.

The module exposes a single public entry-point – `kernel_function` – that
behaves like `torch.abs` but executes the element-wise absolute-value
computation inside a Triton GPU kernel.  The wrapper takes an arbitrary
CUDA tensor, launches the kernel, and returns a tensor with identical
shape & dtype containing `abs(x)`.

Design choices
==============
• The kernel operates on a *contiguous* 1-D view of the data.  Any
  non-contiguous/broadcasted input is first materialised with
  `.contiguous()`.  This keeps the kernel simple and guarantees
  coalesced memory accesses.

• A single generic kernel handles every numeric dtype supported by
  Triton.  The actual element type is inferred from the input pointer,
  and the same piece of code is compiled/separated per dtype by Triton’s
  specialising JIT.

• The computation itself is branch-free and works for signed integral
  and floating point types alike:
        y = tl.where(x < 0, -x, x)

  For booleans (`tl.int1`) the value is already non-negative, so we just
  forward it unchanged.

• Boundary conditions are honoured through a standard predication mask.

The implementation follows the programming guidelines laid out in the
prompt (compile-time constants, proper masking, grid calculation, etc.).
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch


# ----------------------------------------------------------------------------- #
#                              Triton ABS Kernel                                #
# ----------------------------------------------------------------------------- #

@triton.jit
def _abs_kernel(
    x_ptr,                       # *T  – input  tensor
    y_ptr,                       # *T  – output tensor
    n_elements,                  # int – total number of elements
    BLOCK_SIZE: tl.constexpr,    # compile-time block size
):
    """
    Computes `y = abs(x)` for a contiguous vector of length `n_elements`.

    Parameters
    ----------
    x_ptr : *T
        Pointer to the first element of the input tensor.
    y_ptr : *T
        Pointer to the first element of the output tensor.
    n_elements : int
        Total number of elements to process.
    BLOCK_SIZE : tl.constexpr
        Number of elements handled by each Triton program instance
        (must be a power of two for best performance).
    """
    pid = tl.program_id(axis=0)                       # unique program index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # element indices
    mask = offsets < n_elements                       # guard against OOB

    # ---------------------------- load input -------------------------------- #
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # ------------------------- compute absolute value ----------------------- #
    if tl.constexpr(x.dtype == tl.int1):
        # Boolean tensors are already non-negative
        y = x
    else:
        # Works for signed integers & FP types alike
        y = tl.where(x < 0, -x, x)

    # ---------------------------- store result ------------------------------ #
    tl.store(y_ptr + offsets, y, mask=mask)


# ----------------------------------------------------------------------------- #
#                                Public API                                    #
# ----------------------------------------------------------------------------- #

def abs_kernel_impl(inp: torch.Tensor) -> torch.Tensor:
    """
    Element-wise absolute value implemented in Triton.

    Parameters
    ----------
    inp : torch.Tensor
        CUDA tensor of any numeric dtype / shape supported by PyTorch.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape & dtype as `inp` containing `abs(inp)`.
    """
    # ----------------------------- sanity checks ---------------------------- #
    if not isinstance(inp, torch.Tensor):
        raise TypeError("kernel_function expects a torch.Tensor input")
    if not inp.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device")

    # Fast-path for empty tensors – nothing to do
    if inp.numel() == 0:
        return inp.clone()

    # ------------------------------ preparation ----------------------------- #
    # Materialise a contiguous copy for simple 1-D indexing & coalesced loads
    x_contig = inp.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    BLOCK_SIZE = 1024  # good default; power-of-two for better scheduling

    # Triton **grid** – one program per `BLOCK_SIZE` elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # ------------------------------- launch --------------------------------- #
    _abs_kernel[grid](
        x_contig, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape back to the original logical shape (stride differences ignored)
    return out.reshape(inp.shape)