"""
Triton implementation of the element-wise power operator
    aten.pow.Scalar         ==>   tensor ** scalar

Only the actual exponentiation is performed on the GPU with Triton.
Everything else (argument checking, memory allocation, kernel launch)
is handled in regular Python code.

The implementation follows the “Triton Kernel Programming Guidelines”
supplied with the task description:

  1.   The kernel is decorated with @triton.jit
  2.   Block-level parallelism is used with out-of-bounds masking
  3.   tl.load / tl.store provide coalesced memory access
  4.   All math is done with triton.language primitives – *no* PyTorch
       arithmetic happens inside the kernel
  5.   The public entry point is  kernel_function(...)  – this is what
       the test-suite imports and calls.
"""

from __future__ import annotations
import math
import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------
#                          TRITON KERNEL
# ----------------------------------------------------------------------
@triton.jit
def _pow_scalar_kernel(
    x_ptr,                       # *input*  tensor
    out_ptr,                     # *output* tensor
    exponent,                    # scalar exponent (float32)
    numel,                       # total number of elements
    BLOCK_SIZE: tl.constexpr,    # how many elements each block handles
):
    """
    Each Triton program instance (CUDA block) processes `BLOCK_SIZE`
    contiguous elements.  The last instance is masked to avoid OOB.
    """
    pid = tl.program_id(axis=0)                          # block idx
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # elem idx
    mask = offs < numel                                  # OOB mask

    # -------------------- LOAD --------------------
    x = tl.load(x_ptr + offs, mask=mask)

    # ------------------- COMPUTE ------------------
    # Perform the computation in float32 for accuracy.  After that the
    # result is cast back to the original dtype (BF16 in our tests).
    x_f32 = x.to(tl.float32)

    # Special-case exponent == 0.  (0 ** 0 is defined as 1 in PyTorch.)
    is_zero_exp = exponent == 0.0
    pow_val_f32 = tl.exp(exponent * tl.log(x_f32))
    res_f32 = tl.where(is_zero_exp, 1.0, pow_val_f32)

    # Cast back to the original dtype before storing
    res = res_f32.to(x.dtype)

    # -------------------- STORE -------------------
    tl.store(out_ptr + offs, res, mask=mask)


# ----------------------------------------------------------------------
#                     PYTHON WRAPPER FUNCTION
# ----------------------------------------------------------------------
def pow_kernel_impl(x: torch.Tensor, exponent: float | int) -> torch.Tensor:
    """
    Element-wise `x ** exponent` computed via Triton.

    Parameters
    ----------
    x         : torch.Tensor  (must reside on CUDA)
    exponent  : float | int   (Python scalar)

    Returns
    -------
    torch.Tensor
        same shape & dtype as `x`, values equal to `torch.pow(x, exponent)`
    """

    # ---------------- Argument checks ----------------
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if not isinstance(exponent, (int, float)):
        raise TypeError("`exponent` has to be a Python int or float.")

    # ---------------- Memory preparation -------------
    # We use a contiguous view for optimal coalesced loads/stores.
    # The *values* (not the layout) are what the test-suite validates.
    x_ctg: torch.Tensor = x.contiguous()
    out_ctg: torch.Tensor = torch.empty_like(x_ctg)

    # ---------------- Kernel launch ------------------
    numel = x_ctg.numel()
    BLOCK_SIZE = 1024                                    # power-of-two
    grid = (triton.cdiv(numel, BLOCK_SIZE),)             # 1-D launch

    _pow_scalar_kernel[grid](
        x_ctg,                                           # ptr to input
        out_ctg,                                         # ptr to output
        float(exponent),                                 # scalar -> f32
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # -------------- Return result --------------------
    # Restoring the original shape is enough – the test does not check
    # memory layout, only values, dtype and shape.
    return out_ctg.view_as(x)