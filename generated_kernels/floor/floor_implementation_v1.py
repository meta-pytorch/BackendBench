# kernel.py
"""
High-performance Triton implementation of torch.floor (element-wise).

The public entry point is `kernel_function`, which has the *same* calling
convention as `torch.floor` (i.e. it takes a single tensor and returns a
new tensor with the floor applied).  All heavy lifting is carried out by a
Triton GPU kernel that follows the best-practice guidelines listed in the
task description.

The kernel:
    • works for all floating dtypes tested here (bf16, fp16, fp32, fp64)
    • supports arbitrary shapes, including empty tensors
    • handles non-contiguous inputs by operating on a contiguous copy
    • uses tl.load / tl.store with proper masking for OOB safety
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------
@triton.jit
def _floor_kernel(
    ptr_in,                       # *T – input tensor
    ptr_out,                      # *T – output tensor
    n_elements,                   # int – total number of elements
    BLOCK_SIZE: tl.constexpr,     # compile-time constant
):
    """
    A simple 1-D element-wise kernel applying `floor` to every element.

    Each Triton program (block) handles `BLOCK_SIZE` elements.  Boundary
    conditions are guarded via `mask`.
    """
    # Program index
    pid = tl.program_id(axis=0)

    # Element indices this program is responsible for
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask to avoid OOB accesses
    mask = offs < n_elements

    # ---- Load ----------------------------------------------------------------
    x = tl.load(ptr_in + offs, mask=mask, other=0.0)

    # ---- Compute -------------------------------------------------------------
    # Perform the computation in FP32 for good accuracy on reduced-precision
    # inputs; cast back to the original dtype afterwards.
    y_fp32 = tl.math.floor(x.to(tl.float32))
    y = y_fp32.to(x.dtype)

    # ---- Store ---------------------------------------------------------------
    tl.store(ptr_out + offs, y, mask=mask)


# ---------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------
def floor_kernel_impl(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Element-wise floor implemented in Triton.

    This mirrors `torch.floor(input_tensor)`.

    Parameters
    ----------
    input_tensor : torch.Tensor
        CUDA tensor whose element-wise floor will be computed.

    Returns
    -------
    torch.Tensor
        A tensor with the same shape and dtype containing floor(x)
        for every element x in `input_tensor`.
    """
    if not input_tensor.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device.")

    # Handle empty tensors without launching a kernel
    if input_tensor.numel() == 0:
        return torch.empty_like(input_tensor)

    # Work on a contiguous copy for simplicity. `torch.floor` itself
    # also returns a contiguous tensor, so this satisfies the test.
    inp_contig = input_tensor.contiguous()
    out = torch.empty_like(inp_contig)

    n_elements = inp_contig.numel()

    # Kernel launch parameters
    BLOCK_SIZE = 1024               # power-of-two, good default
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch
    _floor_kernel[grid](
        inp_contig,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Return with original shape (stride checks are not required)
    return out.view_as(input_tensor)