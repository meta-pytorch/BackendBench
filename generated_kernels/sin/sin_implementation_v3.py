# kernel.py
"""
A high–performance Triton implementation of the element-wise sine (torch.sin)
operation that is fully compatible with every tensor used by the test-suite.

Main features
-------------
1. Works for every shape – 0-D up to N-D – and for all dtypes supported by the
   test-suite (bf16 / fp16 – it is trivial to extend to fp32 / fp64 / complex).
2. Accepts contiguous **and** non-contiguous inputs.  For simplicity the wrapper
   materialises a *contiguous* copy of the view before launching the kernel
   (this avoids stride bookkeeping inside the GPU code while remaining 100 %
   correct – element order is preserved by `tensor.contiguous()`).
3. Follows Triton best-practices:
      • block size is a compile-time constant (`tl.constexpr`)
      • proper masking for out-of-bounds threads
      • `tl.load` / `tl.store` for memory accesses
4. Keeps numerical work inside Triton – there is **no** fallback to PyTorch
   operations for the actual computation.
"""

import triton
import triton.language as tl
import torch


# ----------------------------------------------------------------------------- #
#                              TRITON KERNEL                                    #
# ----------------------------------------------------------------------------- #
@triton.jit
def _sin_kernel(
    in_ptr,                                    # * Pointer to input data
    out_ptr,                                   # * Pointer to output data
    n_elements,                                # * Number of elements to process
    BLOCK_SIZE: tl.constexpr = 1024,           # * Threads per block (power of 2)
):
    """
    A very small yet efficient element-wise `sin` kernel.

    Each Triton program (CUDA thread-block) handles `BLOCK_SIZE` elements laid
    out consecutively in memory; a final mask keeps threads that run past the
    logical tensor size from reading/writing out-of-bounds.
    """
    # --------------------------------------------------------------------- #
    #                             INDICES                                    #
    # --------------------------------------------------------------------- #
    pid = tl.program_id(axis=0)                         # block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)    # per-thread element id
    mask = offsets < n_elements                         # guard for last block

    # --------------------------------------------------------------------- #
    #                       LOAD  →  COMPUTE  →  STORE                      #
    # --------------------------------------------------------------------- #
    # Load (masked).  `other=0` is ignored where mask==False
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Compute `sin`.
    # `tl.sin` currently operates on fp32, so we cast for accuracy then back.
    x_fp32 = x.to(tl.float32)
    y_fp32 = tl.sin(x_fp32)
    y = y_fp32.to(x.dtype)                              # restore original dtype

    # Write the result
    tl.store(out_ptr + offsets, y, mask=mask)


# ----------------------------------------------------------------------------- #
#                               PYTHON WRAPPER                                  #
# ----------------------------------------------------------------------------- #
def sin_kernel_impl(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Public API – behaves exactly like `torch.sin` but is powered by Triton.

    Parameters
    ----------
    input_tensor : torch.Tensor
        CUDA tensor of arbitrary shape / dtype.

    Returns
    -------
    torch.Tensor
        New tensor with `sin` applied element-wise (same shape & dtype).
    """
    if not input_tensor.is_cuda:
        raise ValueError("`input_tensor` must live on a CUDA device.")

    # Early-exit for empty tensors (legal in PyTorch)
    if input_tensor.numel() == 0:
        return input_tensor.clone()

    # We operate on a contiguous view to simplify indexing inside the kernel.
    # `.contiguous()` creates a *new* tensor that already has the correct element
    # order for the logical indices of the view – this preserves correctness for
    # every strided / transposed / sliced input.
    inp_contig = input_tensor.contiguous()
    out_contig = torch.empty_like(inp_contig)

    # Kernel launch parameters ------------------------------------------------
    n_elements = inp_contig.numel()
    BLOCK_SIZE = 1024                                   # good default (2**k)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)       # 1-D launch grid

    # Execute the Triton kernel
    _sin_kernel[grid](
        inp_contig, out_contig, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape back to the original logical shape (no data movement)
    return out_contig.view(input_tensor.shape)