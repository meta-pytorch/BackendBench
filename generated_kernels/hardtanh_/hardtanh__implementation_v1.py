# kernel.py
#
# Triton implementation of the in-place HardTanh operator
#   y = min(max(x, min_val), max_val)
#
# The public entry-point  `kernel_function`  has the exact
# signature expected by the test-suite:
#
#       kernel_function(input_tensor, min_val, max_val)
#
# • Works in-place  (returned tensor aliases the input)
# • Supports float16 / bfloat16 / float32
# • Handles arbitrary tensor sizes (boundary-safe masking)
# • Uses only Triton ops for the numerical work
#
# -----------------------------------------------------------

import torch
import triton
import triton.language as tl


@triton.jit
def _hardtanh_kernel(ptr_x,                       # *only* tensor pointer
                     numel,                       # total number of elements
                     min_val, max_val,            # scalar bounds
                     BLOCK_SIZE: tl.constexpr):   # compile-time constant
    """
    Simple 1-D elementwise kernel.

    Each Triton "program" (≈ CUDA block) processes BLOCK_SIZE
    contiguous elements.  Masking takes care of the tail that
    falls outside `numel`.
    """
    pid = tl.program_id(axis=0)                   # unique program index
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < numel                        # OOB guard
    x = tl.load(ptr_x + offsets, mask=mask)       # load

    # Clamp to [min_val, max_val]  (all Triton ops!)
    x = tl.maximum(x, min_val)
    x = tl.minimum(x, max_val)

    tl.store(ptr_x + offsets, x, mask=mask)       # write back (in-place)


def hardtanh__kernel_impl(input_tensor: torch.Tensor,
                    min_val: float,
                    max_val: float):
    """
    In-place HardTanh implemented with Triton.

    Parameters
    ----------
    input_tensor : torch.Tensor (CUDA)
        Tensor to be clamped **in place**.
    min_val, max_val : float
        Lower / upper clamp bounds (same semantics as
        `torch.nn.functional.hardtanh_`).

    Returns
    -------
    torch.Tensor
        The *same* tensor object, now containing the clamped data.
    """
    # ----------------- Sanity checks -----------------
    if not input_tensor.is_cuda:
        raise RuntimeError("Triton kernel requires a CUDA tensor.")
    if input_tensor.dtype not in (torch.float16,
                                  torch.bfloat16,
                                  torch.float32):
        raise TypeError(f"Unsupported dtype {input_tensor.dtype}.")
    if min_val > max_val:
        raise ValueError("min_val must be <= max_val.")
    if not input_tensor.is_contiguous():
        # Strided handling is possible but out-of-scope here.
        raise ValueError("Input tensor must be contiguous.")

    numel = input_tensor.numel()
    if numel == 0:                                # nothing to do
        return input_tensor

    # ---------------- Kernel launch -----------------
    BLOCK_SIZE = 1024                             # power of two → good perf
    grid = (triton.cdiv(numel, BLOCK_SIZE),)      # 1-D launch

    _hardtanh_kernel[grid](
        input_tensor,        # ptr_x
        numel,
        min_val, max_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return input_tensor                           # alias – in-place!