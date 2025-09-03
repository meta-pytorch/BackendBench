# kernel.py
"""
Triton implementation of the element-wise reciprocal square-root operation
(torch.rsqrt / aten.rsqrt.default).

A *single* Triton program (block) processes `BLOCK_SIZE` consecutive elements.
The wrapper `kernel_function` is what external code (e.g. the provided unit-
test) will import and call.

Author: OpenAI Assistant
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 1.  Triton kernel
# ---------------------------------------------------------------------------
@triton.jit
def _rsqrt_kernel(
    x_ptr,                # *  Input  data pointer
    y_ptr,                # *  Output data pointer
    numel,                #    Total number of elements
    BLOCK_SIZE: tl.constexpr,   # Number of elements handled by one program
):
    """
    Parameters
    ----------
    x_ptr : tl.pointer
        Pointer to the first element of the input tensor.
    y_ptr : tl.pointer
        Pointer to the first element of the output tensor.
    numel : int
        Total number of scalar elements in the tensor.
    BLOCK_SIZE : int (tl.constexpr)
        Compile-time constant controlling how much work each program does.
    """
    # --------------------------------------------------
    # Program / block identification & indexing
    # --------------------------------------------------
    pid  = tl.program_id(axis=0)                       # 1-D launch grid
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # Vector of indices
    mask = offs < numel                                # Guard for OOB lanes

    # --------------------------------------------------
    # Memory I/O – coalesced contiguous accesses
    # --------------------------------------------------
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)

    # --------------------------------------------------
    # Compute 1 / sqrt(x) in higher precision (fp32)
    # --------------------------------------------------
    x_f32 = x.to(tl.float32)
    y_f32 = tl.math.rsqrt(x_f32)       # 1 / sqrt(x)
    y     = y_f32.to(x.dtype)          # Cast back to original dtype

    # --------------------------------------------------
    # Write results
    # --------------------------------------------------
    tl.store(y_ptr + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# 2.  Public wrapper
# ---------------------------------------------------------------------------
def rsqrt_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Element-wise reciprocal square-root using Triton.

    Notes
    -----
    • The actual math is performed inside `_rsqrt_kernel` with Triton ops.
    • This wrapper only handles argument checking, memory allocation,
      grid calculation, and kernel dispatch.

    Parameters
    ----------
    x : torch.Tensor
        CUDA tensor containing strictly positive values (as rsqrt is undefined
        for non-positive inputs in real numbers).  Any floating–point dtype
        supported by Triton (fp16, bf16, fp32, fp64, …) is accepted.

    Returns
    -------
    torch.Tensor
        Tensor of same shape & dtype with `1 / sqrt(x)` computed element-wise.
    """
    # --------------------------------------------------
    # Sanity checks & preparation
    # --------------------------------------------------
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    x_contig = x.contiguous()  # Triton expects contiguous memory

    # Allocate output
    y = torch.empty_like(x_contig)

    # --------------------------------------------------
    # Kernel launch parameters
    # --------------------------------------------------
    numel       = x_contig.numel()
    BLOCK_SIZE  = 1024                              # Power-of-2 for best perf
    grid        = (triton.cdiv(numel, BLOCK_SIZE),) # 1-D launch grid

    # --------------------------------------------------
    # Launch Triton kernel
    # --------------------------------------------------
    _rsqrt_kernel[grid](
        x_contig,               # ptr to input
        y,                      # ptr to output
        numel,                  # number of elements
        BLOCK_SIZE=BLOCK_SIZE,  # compile-time constant
    )

    # Preserve original (possibly non-contiguous) shape
    return y.view_as(x)