# kernel.py
"""
Triton implementation of the element-wise exponential (torch.exp).

The file exposes a single public entry point – ``kernel_function`` – whose
Python signature is intentionally identical to ``torch.exp`` (one tensor
argument, same return type).  Internally the heavy lifting is performed by a
Triton GPU kernel that:

• Works on *flat* 1-D views of the input (arbitrary shapes are supported by
  flattening then re-viewing the result).
• Handles all boundary conditions via masking.
• Supports the most common floating dtypes used with GPUs
  (float16 / bfloat16 / float32 / float64).
• Never calls any PyTorch math routines inside the kernel – everything is
  implemented with `triton.language` ops.

The implementation follows the “Triton Kernel Programming Guidelines” shipped
with the task statement.
"""

import triton
import triton.language as tl
import torch


# -----------------------------------------------------------------------------
# Triton kernel
# -----------------------------------------------------------------------------
@triton.jit
def _exp_kernel(
    in_ptr,                  # *void        – pointer to input  tensor
    out_ptr,                 # *void        – pointer to output tensor
    n_elements,              # int32 / int64 – total #elements (flattened)
    BLOCK_SIZE: tl.constexpr # compile-time  – how many elements per block
):
    """
    A single-dimensional grid where each program instance (thread-block)
    processes ``BLOCK_SIZE`` consecutive elements.

    Memory accesses:
        • Fully coalesced for contiguous tensors because the kernel walks the
          flattened storage in order.
        • Boundary conditions are handled via a mask.
    """

    # --------------------------------------------
    # Compute the range this program instance owns
    # --------------------------------------------
    pid = tl.program_id(axis=0)              # current block id
    block_start = pid * BLOCK_SIZE           # first element this block handles
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # [block_start, …, +BS-1]
    mask = offsets < n_elements              # OOB mask for last block

    # --------------------------------------------
    # Load -> Compute -> Store (element-wise exp)
    # --------------------------------------------
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Compute in fp32 for accuracy, then cast back **inside the kernel** so the
    # *returned tensor dtype* exactly matches the input dtype.
    y_fp32 = tl.exp(x.to(tl.float32))
    y = y_fp32.to(x.dtype)

    tl.store(out_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------------------
# Python wrapper – this is what the test-suite imports
# -----------------------------------------------------------------------------
def exp_kernel_impl(inp: torch.Tensor) -> torch.Tensor:
    """
    Element-wise exponential, powered by Triton.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor living on a CUDA device.  Must be of floating dtype
        supported by Triton (fp16 / bf16 / fp32 / fp64).

    Returns
    -------
    torch.Tensor
        Same shape & dtype as ``inp`` where each element is ``exp(inp[i])``.
    """
    # ----------------------------  Safety checks  ---------------------------
    if not inp.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device.")

    if inp.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise TypeError(
            f"Unsupported dtype {inp.dtype}. "
            "Supported: float16, bfloat16, float32, float64."
        )

    # Triton kernels are fastest on contiguous memory – fall back to a
    # contiguous *view* (no data copy) when possible, otherwise clone.
    if not inp.is_contiguous():
        inp_contig = inp.contiguous()
    else:
        inp_contig = inp

    # Allocate output tensor (same dtype / device)
    out = torch.empty_like(inp_contig)

    # ----------------------------  Launch kernel  ---------------------------
    n_elements = inp_contig.numel()
    BLOCK_SIZE = 1024  # good default (multiple of 32, ≤ 1024)

    # 1-D launch grid: one program per `BLOCK_SIZE` elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _exp_kernel[grid](
        inp_contig,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,     # balanced for 1K element blocks
        num_stages=2,    # pipeline depth
    )

    # If we had to make a contiguous copy for the kernel, reshape back so the
    # *logical* shape matches the original tensor (strides might differ).
    return out.view_as(inp)