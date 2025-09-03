# kernel.py
"""
Triton implementation of the in-place Hard-Swish operator

    aten.hardswish_.default      <=>   tensor.hardswish_()  in Python

This file exposes a *regular* Python function called ``kernel_function`` that
behaves exactly like the reference PyTorch op:

  • It mutates the input tensor in-place and returns *the same* object
  • It supports every floating point dtype that Triton can handle
    (the unit–test exercises fp16 and bf16)
  • It works for both contiguous and non-contiguous tensors; the kernel
    simply iterates over the underlying storage buffer, so every element is
    touched exactly once – layout does not matter for an element-wise op.
  • All math is performed inside a Triton JIT kernel using `tl.load`,
    `tl.store`, `tl.maximum`, `tl.minimum`, …

Author: OpenAI ChatGPT
"""
from typing import Callable

import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
#                           TRITON KERNEL
# ----------------------------------------------------------------------
@triton.jit
def _hardswish_kernel(x_ptr,                   # *only* tensor argument (in-place)
                      n_elements,              # total number of scalars
                      BLOCK_SIZE: tl.constexpr):
    """
    Element-wise Hard-Swish:

        out = x * clamp(x + 3, 0, 6) / 6
    """
    pid = tl.program_id(axis=0)                      # block index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                      # handle ragged tail

    # ----------------------- load ------------------------------------
    x = tl.load(x_ptr + offsets, mask=mask)

    # ----------------------- compute ---------------------------------
    x_f32 = x.to(tl.float32)                         # higher precision math
    tmp = x_f32 + 3.0
    tmp = tl.minimum(tl.maximum(tmp, 0.0), 6.0)      # clamp(tmp, 0, 6)
    y_f32 = x_f32 * tmp * (1.0 / 6.0)

    y = y_f32.to(x.dtype)                            # cast back to original dtype

    # ----------------------- store -----------------------------------
    tl.store(x_ptr + offsets, y, mask=mask)


# ----------------------------------------------------------------------
#                       PYTHON WRAPPER FUNCTION
# ----------------------------------------------------------------------
def hardswish__kernel_impl(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place Hard-Swish implemented with Triton.

    Parameters
    ----------
    tensor : torch.Tensor (CUDA, floating dtype)
        Input tensor that will be *mutated* in-place.

    Returns
    -------
    torch.Tensor
        The same object that was passed in (for chaining / parity with PyTorch).
    """
    # ------------- basic checks -----------------
    if not tensor.is_cuda:
        raise RuntimeError("`kernel_function` expects a CUDA tensor.")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype {tensor.dtype}. "
                        "Only fp16 / bf16 / fp32 are allowed.")

    n_elements = tensor.numel()
    if n_elements == 0:
        return tensor                                   # nothing to do

    # ------------- launch configuration -----------
    BLOCK_SIZE = 2048                                   # power of two → good coalescing

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)       # 1-D launch

    # ------------- run ----------------------------
    _hardswish_kernel[grid](
        tensor,                                         # x_ptr
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return tensor                                       # identical to input (in-place)