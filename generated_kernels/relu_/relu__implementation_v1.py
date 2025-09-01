# kernel.py
"""
Triton implementation of the in-place ReLU operator (``aten.relu_.default``).

The public entry point is ``kernel_function``.  It has *exactly* the same
signature and semantics as ``torch.relu_``:

    >>> out = kernel_function(x)      # modifies ``x`` in-place
    >>> out is x                      # True

Only the core element-wise computation is performed inside a Triton kernel,
strictly following the guidelines supplied with the task.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
#                               TRITON KERNEL
# -----------------------------------------------------------------------------
@triton.jit
def _relu_inplace_kernel(
    ptr_x,                       # *base* pointer to the tensor data
    num_elements,                # total number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,    # elements processed by a single program
):
    """
    Simple, generic 1-D in-place ReLU.

    Each *program* (CUDA thread-block) processes exactly ``BLOCK_SIZE`` scalar
    elements.  All tensor layouts (contiguous or not) map to a flat array
    because we operate on the underlying storage.
    """
    pid = tl.program_id(axis=0)                           # unique block index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)      # [0 .. BLOCK_SIZE)

    # Guard against out-of-range accesses (tail blocks)
    mask = offsets < num_elements

    x = tl.load(ptr_x + offsets, mask=mask, other=0)      # load
    y = tl.where(x > 0, x, 0)                             # ReLU
    tl.store(ptr_x + offsets, y, mask=mask)               # store in-place


# -----------------------------------------------------------------------------
#                      PYTHON WRAPPER (PUBLIC ENTRY POINT)
# -----------------------------------------------------------------------------
def relu__kernel_impl(tensor: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for ``torch.relu_`` implemented with Triton.

    Parameters
    ----------
    tensor : torch.Tensor
        Any floating-point CUDA tensor (f16 / bf16 are tested).  It is mutated
        *in-place*; the function returns the **same** object.

    Returns
    -------
    torch.Tensor
        The input tensor (now containing ReLU-ed data).
    """
    # --------------------------- sanity checks ------------------------------
    if not tensor.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device.")
    if not tensor.is_floating_point():
        raise TypeError("ReLU is only defined for floating-point tensors.")
    # Nothing to do for empty tensors
    numel: int = tensor.numel()
    if numel == 0:
        return tensor

    # --------------------------- kernel launch ------------------------------
    BLOCK_SIZE = 1024                           # power-of-two for good measure
    grid = (triton.cdiv(numel, BLOCK_SIZE),)    # 1-D launch

    _relu_inplace_kernel[grid](                 # noqa: SLF001
        tensor,                                 # base pointer
        numel,                                  # total elements
        BLOCK_SIZE=BLOCK_SIZE,                  # meta-parameter
    )

    return tensor  # in-place semantics: return *same* object


__all__ = ["kernel_function"]