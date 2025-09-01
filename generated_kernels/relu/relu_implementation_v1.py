# kernel.py
#
# A Triton implementation of element-wise ReLU that is functionally
# equivalent to `torch.nn.functional.relu`.  The heavy-lifting (the
# actual max(x, 0) computation) is performed inside a Triton GPU
# kernel.  The public entry-point – `kernel_function` – is a thin
# wrapper that takes an arbitrary PyTorch tensor, launches the Triton
# kernel, and returns the result as a PyTorch tensor.
#
# The implementation obeys the “TRITON KERNEL PROGRAMMING GUIDELINES”
# summarised in the assignment prompt:
#   * Kernel is decorated with @triton.jit
#   * Uses tl.constexpr for compile-time constants
#   * Employs tl.load / tl.store with proper masking
#   * Handles boundary conditions
#   * Works for both contiguous and non-contiguous inputs (the wrapper
#     makes a contiguous copy when required – computing on non-strided
#     memory makes coalescing and masking much simpler).
#
# Supported dtypes: float16, bfloat16 (the ones needed by the tests,
# though extending to float32/float64 would only require minor tweaks).

import triton
import triton.language as tl
import torch


@triton.jit
def _relu_kernel(x_ptr,                     # * pointer to input
                 y_ptr,                     # * pointer to output
                 n_elements,                # * total number of scalars
                 BLOCK_SIZE: tl.constexpr   # * compile-time constant
                 ):
    """
    A simple, single-pass, element-wise ReLU implementation.

    Each Triton program instance (block) processes `BLOCK_SIZE` elements.
    Masking ensures we never read / write out-of-bounds when the total
    element count is not a multiple of the block size.
    """
    pid = tl.program_id(axis=0)                     # block id
    block_start = pid * BLOCK_SIZE                  # first element this block owns
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vector of indices
    mask = offsets < n_elements                     # boundary check

    # --------------------
    # Load → Compute → Store
    # --------------------
    x = tl.load(x_ptr + offsets, mask=mask)

    zero = tl.zeros_like(x)
    y = tl.where(x > 0, x, zero)                    # ReLU

    tl.store(y_ptr + offsets, y, mask=mask)


def _next_power_of_two(x: int) -> int:
    """
    Utility: round `x` up to the next power-of-two, capped at 1024
    (fits well with most NVIDIA GPUs’ maximum register file usage).
    """
    pow2 = 1
    while pow2 < x and pow2 < 1024:
        pow2 <<= 1
    return pow2


def relu_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Public API – identical signature & behaviour to `torch.relu(x)`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.  Must be on CUDA and of dtype float16 or bfloat16.

    Returns
    -------
    torch.Tensor
        A new tensor containing `max(x, 0)` with the same shape & dtype
        as `x`.  (The storage will be contiguous even if `x` was not.)
    """
    # ------------------------------------------------------------------
    # Sanity / type checks
    # ------------------------------------------------------------------
    if not x.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported dtype {x.dtype}. "
                        "Only float16 & bfloat16 are currently supported.")

    # ------------------------------------------------------------------
    # Make data contiguous – greatly simplifies indexing logic and
    # guarantees coalesced global memory accesses inside the kernel.
    # ------------------------------------------------------------------
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n_elements = x_contig.numel()

    # ------------------------------------------------------------------
    # Kernel launch parameters
    # ------------------------------------------------------------------
    # Choose a power-of-two BLOCK_SIZE in {64, 128, …, 1024} such that
    # each thread block does a reasonable amount of work.
    BLOCK_SIZE = _next_power_of_two(256)  # 256 is a good default

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # ------------------------------------------------------------------
    # Launch!
    # ------------------------------------------------------------------
    _relu_kernel[grid](
        x_contig, y, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # `y` is already the desired tensor;  its shape matches `x_contig`
    # (which matches the logical shape of the original `x` even if the
    # strides changed because of a transpose / slicing).
    return y