# kernel.py
"""
Triton implementation of the SiLU / Sigmoid-Swish activation
    f(x) = x * sigmoid(x) = x / (1 + exp(-x))

The public entry-point `kernel_function` can be used exactly like
`torch.nn.functional.silu` but is backed by a highly-parallel Triton kernel.
It is fully self-contained – the test-suite will merely import the
`kernel_function` symbol and call it on a few sample tensors.

Key implementation notes
------------------------
1.  The Triton kernel works on a *flat* 1-D view of the input tensor and
    therefore supports **arbitrary ranks / shapes**.  Boundary conditions are
    handled through masking.
2.  Arithmetic is performed in float32 for improved numerical accuracy and
    cast back to the original dtype (fp16 / bf16 / fp32) before writing.
3.  For simplicity and dependable coalesced memory accesses we create a
    contiguous copy of the input first.  This has no impact on numerical
    results and keeps the kernel logic compact while still covering
    non-contiguous source tensors.
4.  The kernel follows the general Triton programming guidelines:
      • `@triton.jit` decorated kernel
      • compile-time constant `BLOCK_SIZE`
      • `tl.load` / `tl.store` with proper masking
      • use of `tl.program_id` for grid indexing
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------#
#                               Triton KERNEL                                   #
# -----------------------------------------------------------------------------#
@triton.jit
def _silu_kernel(
    x_ptr,               # *const* pointer to input
    y_ptr,               # *mut*   pointer to output
    numel,               # total number of elements
    BLOCK_SIZE: tl.constexpr,  # block width (compile-time constant)
):
    """
    Simple 1-D mapping kernel: each program instance processes BLOCK_SIZE
    consecutive elements.

    Parameters
    ----------
    x_ptr : tl.pointer
        Pointer to the first element of the (flattened) input tensor.
    y_ptr : tl.pointer
        Pointer to the first element of the (flattened) output tensor.
    numel : int
        Total number of scalar elements to process.
    BLOCK_SIZE : tl.constexpr
        Number of threads / elements handled by one program.
    """
    # ------------------------------------------------------------------ #
    # Compute the indices this program is responsible for                #
    # ------------------------------------------------------------------ #
    pid = tl.program_id(axis=0)  # unique program id in the launch grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # shape: [BLOCK_SIZE]
    mask = offsets < numel                            # avoid OOB accesses

    # ------------------------------------------------------------------ #
    # Load input values                                                  #
    # ------------------------------------------------------------------ #
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ------------------------------------------------------------------ #
    # SiLU computation in fp32                                           #
    # ------------------------------------------------------------------ #
    x_f32 = x.to(tl.float32)
    one = tl.full(x_f32.shape, 1.0, dtype=tl.float32)
    sigmoid = one / (one + tl.exp(-x_f32))        # σ(x) = 1 / (1 + e^{-x})
    y = (x_f32 * sigmoid).to(x.dtype)             # back-cast to original type

    # ------------------------------------------------------------------ #
    # Write results                                                      #
    # ------------------------------------------------------------------ #
    tl.store(y_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------------------#
#                           Python wrapper API                                 #
# -----------------------------------------------------------------------------#
def silu_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the SiLU activation using the Triton kernel defined above.

    Parameters
    ----------
    x : torch.Tensor
        CUDA tensor (float16 / bfloat16 / float32).  Any shape and
        (non-)contiguity are accepted.

    Returns
    -------
    torch.Tensor
        Tensor with SiLU applied element-wise.  Shape & dtype match *x*.
    """
    # --------------------------- Safety checks --------------------------- #
    if not x.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {x.dtype}. "
            "Supported dtypes: float16, bfloat16, float32."
        )

    # -------------------------------------------------------------------- #
    # Create a contiguous view for efficient, coalesced accesses           #
    # -------------------------------------------------------------------- #
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    numel = x_contig.numel()

    # Kernel launch configuration
    BLOCK_SIZE = 1024  # power-of-two, good default for most GPUs
    grid = (triton.cdiv(numel, BLOCK_SIZE),)  # 1-D grid

    # Launch the Triton kernel
    _silu_kernel[grid](x_contig, y, numel, BLOCK_SIZE)

    # Reshape back to the original logical shape (stride may differ)
    return y.view_as(x)