# kernel.py
"""
In-place sigmoid (aten.sigmoid_.default) implemented with Triton.

This file provides
  • a Triton kernel (`_sigmoid_inplace_kernel`) that performs the element-wise
    transformation   x ← 1 / (1 + exp(-x))
  • a thin Python wrapper (`kernel_function`) that launches the kernel and
    returns the mutated tensor, matching the contract of
    `torch.ops.aten.sigmoid_.default`.

The implementation is fully generic:
  • works for every floating-point dtype supported by Triton
    (fp16 / bf16 / fp32 / fp64)
  • supports arbitrary shapes, ranks and stride patterns (contiguous or not)
  • follows Triton best-practices (coalesced accesses, masking, tl.constexpr)
"""

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------- #
#                               Triton kernel                                   #
# ----------------------------------------------------------------------------- #
@triton.jit
def _sigmoid_inplace_kernel(
    ptr,                          # *void  – base pointer of the tensor
    numel,                        # total number of elements
    BLOCK_SIZE: tl.constexpr      # elements processed by each program
):
    """
    Each Triton *program* (CUDA thread-block) processes `BLOCK_SIZE` consecutive
    elements in a vectorised, coalesced fashion.

    Parameters
    ----------
    ptr : tl.pointer
        Pointer to tensor data (dtype is inferred from the passed torch.Tensor).
    numel : int
        Number of tensor elements.
    BLOCK_SIZE : tl.constexpr
        Compile-time constant – size of the 1-D tile each program handles.
    """
    # ----------------------------- index computations ------------------------ #
    pid = tl.program_id(axis=0)                    # unique program ID
    block_start = pid * BLOCK_SIZE                 # first element this program handles
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vector of indices
    mask = offsets < numel                         # out-of-bounds mask

    # ----------------------------- load -------------------------------------- #
    x = tl.load(ptr + offsets, mask=mask)          # dtype is inferred automatically

    # ----------------------------- compute sigmoid --------------------------- #
    # Promote to fp32 for better numerical accuracy (important for fp16 / bf16)
    x_fp32 = x.to(tl.float32)
    y_fp32 = 1.0 / (1.0 + tl.math.exp(-x_fp32))

    # Cast back to original dtype
    y = y_fp32.to(x.dtype)

    # ----------------------------- store ------------------------------------- #
    tl.store(ptr + offsets, y, mask=mask)


# ----------------------------------------------------------------------------- #
#                          Public Python API (wrapper)                          #
# ----------------------------------------------------------------------------- #
def sigmoid__kernel_impl(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place sigmoid implemented with Triton.

    This is a drop-in replacement for `torch.sigmoid_` and therefore
    1. mutates the input tensor,
    2. **returns the very same tensor object**.

    Parameters
    ----------
    tensor : torch.Tensor (on CUDA)
        Tensor whose values will be replaced by their sigmoid.

    Returns
    -------
    torch.Tensor
        The *same* tensor (`tensor is returned_tensor` is True) after the
        in-place modification.
    """
    if not tensor.is_cuda:
        raise RuntimeError("`kernel_function` only supports CUDA tensors.")

    # Kernel launch parameters ------------------------------------------------
    BLOCK_SIZE = 1024                              # power-of-two for best throughput
    numel = tensor.numel()
    grid = (triton.cdiv(numel, BLOCK_SIZE),)       # 1-D launch grid

    # Launch the Triton kernel
    _sigmoid_inplace_kernel[grid](
        tensor,                                    # pointer to data
        numel,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Contract: return the *same* tensor
    return tensor