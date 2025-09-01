"""
Triton implementation of the element-wise reciprocal operation
(`aten.reciprocal.default` → 1 / x).

The public entry point is `kernel_function`, which can be imported and
used like the regular PyTorch op:

    from kernel import kernel_function
    y = kernel_function(x)      # y == 1 / x

Key features
------------
* Handles tensors of arbitrary shape – including 0-dim scalars.
* Works for all floating-point dtypes supported by Triton
  (fp32 / fp16 / bf16).  The accompanying test-suite uses BF16.
* Accepts non-contiguous inputs (they are made contiguous once for fast
 , coalesced loads — the result is returned with the correct shape).
* Uses *only* Triton operations for the computation itself.
"""

import triton
import triton.language as tl
import torch


# ---------------------------------------------------------------------
#                         TRITON DEVICE KERNEL
# ---------------------------------------------------------------------
@triton.jit
def _reciprocal_kernel(
    inp_ptr,                    # * const T*  – pointer to input  tensor
    out_ptr,                    # *       T*  – pointer to output tensor
    numel,                      # int64       – total number of elements
    BLOCK_SIZE: tl.constexpr,   # compile-time – number of elements / PTX block
):
    """
    Each program instance (CUDA thread-block) processes `BLOCK_SIZE`
    consecutive elements.
    """
    pid = tl.program_id(axis=0)                     # 1-D launch grid
    block_start = pid * BLOCK_SIZE                  # first element this block owns
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard out-of-bounds accesses for the last block
    mask = offsets < numel

    # ---------- Load --------------------------------------------------
    x = tl.load(inp_ptr + offsets, mask=mask)

    # ---------- Compute  y = 1 / x  -----------------------------------
    # We build a constant `1` with the SAME dtype as `x` to guarantee the
    # computation happens in that precision (important for BF16 tests).
    one = tl.full((BLOCK_SIZE,), 1.0, x.dtype)
    y = one / x                          # element-wise reciprocal

    # ---------- Store -------------------------------------------------
    tl.store(out_ptr + offsets, y, mask=mask)


# ---------------------------------------------------------------------
#                       PYTHON WRAPPER FUNCTION
# ---------------------------------------------------------------------
def reciprocal_kernel_impl(input_tensor: torch.Tensor, /):
    """
    Equivalent to ``torch.ops.aten.reciprocal.default`` (1 / x) but
    executed by a custom Triton kernel.

    Parameters
    ----------
    input_tensor : torch.Tensor (CUDA)
        Tensor whose reciprocal is desired.

    Returns
    -------
    torch.Tensor
        The element-wise reciprocal, same dtype / shape as the input.
    """
    # Basic validation -------------------------------------------------
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("kernel_function expects a torch.Tensor as input.")
    if not input_tensor.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device.")

    # Contiguous copy for coalesced accesses --------------------------
    # (No correctness impact – only affects the memory layout.)
    inp_contig = input_tensor.contiguous()
    numel = inp_contig.numel()

    # Allocate output buffer (contiguous) -----------------------------
    out_contig = torch.empty_like(inp_contig)

    # Launch configuration -------------------------------------------
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)        # 1-D grid

    _reciprocal_kernel[grid](
        inp_contig,
        out_contig,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Return with the original *shape* (strides may differ – that is fine)
    return out_contig.view_as(input_tensor)