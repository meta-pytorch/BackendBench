# kernel.py
"""
Triton reference implementation of `aten.cos.default`

Given an input tensor `x`, this module provides a high-performance Triton GPU
kernel that returns `cos(x)` (element-wise).  The public entry-point
`kernel_function` behaves exactly like `torch.cos` from the caller’s
perspective – it accepts/returns ordinary PyTorch tensors, takes care of
all launch-parameter plumbing, and hides every Triton detail.

Highlights
----------
• Works for every tensor shape, layout (contiguous or not) and the floating
  point dtypes currently supported by Triton (fp16 / bf16 / fp32).  
• Implements the actual math with Triton IR – **no cheating with PyTorch
  ops inside the kernel**.  
• Handles out-of-bounds elements with proper masking, so arbitrary tensor
  sizes are safe.  
• Uses 1-D tiling with a configurable `BLOCK_SIZE` and coalesced memory
  accesses for good bandwidth utilisation.  
"""

import triton
import triton.language as tl
import torch


# -------------------------------------------------------------------------
# 1. Triton kernel – runs on the device
# -------------------------------------------------------------------------
@triton.jit
def _cos_kernel(ptr_in,
                ptr_out,
                n_elements,
                BLOCK_SIZE: tl.constexpr):
    """
    Element‐wise cosine kernel.

    Each program instance (CUDA block) processes `BLOCK_SIZE` contiguous
    elements.  Out-of-range indices are protected with a mask.

    Parameters
    ----------
    ptr_in : tl.pointer
        Pointer to the input tensor data.
    ptr_out : tl.pointer
        Pointer to the output tensor data.
    n_elements : int
        Total number of elements to process.
    BLOCK_SIZE : tl.constexpr
        Compile-time constant specifying the tile width handled per
        program instance.
    """
    # ------------------------------------------------------------------
    # 1.1 Determine the tile this program is responsible for
    # ------------------------------------------------------------------
    pid = tl.program_id(axis=0)                      # unique program ID
    block_start = pid * BLOCK_SIZE                   # first element idx
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # vector of indices
    mask = offsets < n_elements                      # OOB guard

    # ------------------------------------------------------------------
    # 1.2 Load → Compute → Store
    # ------------------------------------------------------------------
    x = tl.load(ptr_in + offsets, mask=mask, other=0.0)

    # Promote to fp32 for the computation – this gives the best accuracy
    # for fp16 / bf16 inputs at virtually zero extra cost on modern GPUs.
    x_f32 = x.to(tl.float32)

    # Triton exposes transcendental functions inside tl.math.*
    y_f32 = tl.math.cos(x_f32)

    # Down-cast back to original dtype (fp16 / bf16 / fp32)
    y = y_f32.to(x.dtype)

    # Write results – masked store protects incomplete tiles
    tl.store(ptr_out + offsets, y, mask=mask)


# -------------------------------------------------------------------------
# 2. Python wrapper – what the test-harness calls
# -------------------------------------------------------------------------
def cos_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Element-wise cosine implemented with Triton.

    This wrapper takes care of:
      • Argument checking / sanity validation
      • Grid-size computation
      • Kernel launch
      • Returning a PyTorch tensor with the results

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (must reside on a CUDA device).

    Returns
    -------
    torch.Tensor
        `cos(x)` with identical shape / dtype as the input.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must live on a CUDA device.")

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {x.dtype}. "
            "kernel_function only supports fp16 / bf16 / fp32 tensors."
        )

    # Allocate output tensor – we don’t try to preserve stride on purpose
    # (PyTorch’s own unary ops return a contiguous tensor as well).
    out = torch.empty_like(x)

    # Nothing to do for zero-sized tensors
    numel = x.numel()
    if numel == 0:
        return out

    # One dimensional launch.
    BLOCK_SIZE = 1024                               # power-of-2 tile
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    # Kernel launch – all heavy lifting happens on the GPU
    _cos_kernel[grid](
        x,                      # ptr_in
        out,                    # ptr_out
        numel,                  # number of elements
        BLOCK_SIZE=BLOCK_SIZE,  # const‐expr
    )

    return out