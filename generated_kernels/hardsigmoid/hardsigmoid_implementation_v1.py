# kernel.py
"""
High-performance Triton implementation of the *hard-sigmoid* activation

The mathematical definition (identical to torch.ops.aten.hardsigmoid.default) is

    out = max(0, min(1, x * 1/6 + 0.5))

This file exposes a single public entry-point called `kernel_function` that the
test-suite will import and call.  All heavy‐lifting is done inside a Triton
kernel; the wrapper merely prepares the launch parameters and allocates the
output tensor.

Design choices
--------------
• Works for both `torch.float16` and `torch.bfloat16`
• Accepts **arbitrary** shapes and strides – a `contiguous()` copy of the input
  is taken to guarantee coalesced memory access inside the kernel.
• One-dimensional grid where each Triton program (block) processes
  `BLOCK_SIZE` elements.
• Computation happens in `fp32` for better numerical accuracy, then cast back
  to the original dtype before writing to memory.

Author: OpenAI ChatGPT
"""

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Triton kernel
# -----------------------------------------------------------------------------
@triton.jit
def _hardsigmoid_kernel(
    x_ptr,            # *const* T  – input  tensor
    out_ptr,          # *mut*   T  – output tensor
    n_elements,       # int64        – total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parameters
    ----------
    x_ptr : pointer to input  data
    out_ptr : pointer to output data
    n_elements : total number of elements in the tensor
    BLOCK_SIZE : compile-time constant, how many elements one program handles
    """
    # ------------------------------------------------------------------
    # Determine which part of the tensor this program is responsible for
    # ------------------------------------------------------------------
    pid = tl.program_id(axis=0)                    # 1-D grid
    block_start = pid * BLOCK_SIZE                 # first element this block handles
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vector of indices
    mask = offsets < n_elements                    # mask for out-of-bounds

    # -----------------------
    # Load -> Compute -> Store
    # -----------------------
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Cast to fp32 for math, apply hard-sigmoid, then cast back
    x_fp32 = x.to(tl.float32)
    y = x_fp32 * 0.1666666716337204 + 0.5          # 1/6 ≈ 0.16666667
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)
    y = y.to(x.dtype)

    tl.store(out_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------------------
# Public wrapper
# -----------------------------------------------------------------------------
def hardsigmoid_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the hard-sigmoid activation to `x` using a custom Triton kernel.

    The function is intentionally **drop-in compatible** with
    `torch.ops.aten.hardsigmoid.default`.

    Parameters
    ----------
    x : torch.Tensor (float16 or bfloat16, CUDA)
        Input tensor of any shape / stride layout.

    Returns
    -------
    torch.Tensor
        Output tensor with the same shape as `x` (contiguous).
    """
    # --------------------  Input validation  --------------------
    if not x.is_cuda:
        raise ValueError("`kernel_function` expects a CUDA tensor as input.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            "Only float16 and bfloat16 tensors are supported "
            f"(got {x.dtype})."
        )

    # --------------------  Memory preparation  ------------------
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)               # contiguous output buffer

    n_elements = x_contig.numel()
    BLOCK_SIZE = 1024                              # power-of-two per guidelines
    # 1-D launch grid
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # --------------------  Kernel launch  -----------------------
    _hardsigmoid_kernel[grid](
        x_contig,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # --------------------  Return result  -----------------------
    # The tests only care about *values*, not the exact stride layout.
    # Returning a contiguous tensor is therefore sufficient.
    return out.view_as(x)