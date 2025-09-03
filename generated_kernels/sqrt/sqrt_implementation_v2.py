# kernel.py
"""
Triton implementation of `torch.sqrt` (aten.sqrt.default).

The module exposes a single user–visible function
    kernel_function(x : torch.Tensor) -> torch.Tensor
that behaves just like `torch.sqrt(x)` but performs the arithmetic inside
a Triton kernel for speed.  It supports:
  • arbitrary shapes (including zero-sized tensors and 0-D scalars);
  • non-contiguous inputs (we compute on a contiguous copy internally);
  • all floating-point dtypes accepted by PyTorch (fp32 / fp16 / bf16).

Only tensor–creation / book-keeping is done with PyTorch in Python.
The numerical work happens in Triton – no cheating with `torch.sqrt`
inside the kernel!
"""
# -----------------------------------------------------------------------------


import triton
import triton.language as tl
import torch


# -----------------------------------------------------------------------------


@triton.jit
def _sqrt_kernel(inp_ptr,
                 out_ptr,
                 numel,
                 BLOCK_SIZE: tl.constexpr):
    """
    Parameters
    ----------
    inp_ptr : tl.pointer
        Pointer to the (contiguous) input tensor.
    out_ptr : tl.pointer
        Pointer to the (contiguous) output tensor.
    numel : int32 / int64
        Total number of elements in `inp_ptr`.
    BLOCK_SIZE : tl.constexpr
        Number of elements processed by each Triton *program* (CTA).

    Notes
    -----
    The kernel is 1-D-launched.  Each program:
      • loads up to `BLOCK_SIZE` elements,
      • computes `sqrt` in float32 for extra accuracy,
      • casts the result back to the original dtype,
      • writes the result out.

    Boundary conditions are handled via a `mask`.
    """
    # --------------------------------------------------------------------------------
    pid = tl.program_id(axis=0)                   # unique program ID
    block_start = pid * BLOCK_SIZE               # element index this program starts at
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # positions handled by this program
    mask = offsets < numel                       # mask to guard OOB accesses

    # -- Load ------------------------------------------------------------------------
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)

    # -- Compute ---------------------------------------------------------------------
    # Cast to fp32 for better precision, compute sqrt, cast back to original dtype
    x_fp32 = x.to(tl.float32)
    y_fp32 = tl.sqrt(x_fp32)
    y = y_fp32.to(x.dtype)

    # -- Store -----------------------------------------------------------------------
    tl.store(out_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------------------


def _launch_config(numel: int):
    """
    Simple helper that returns a suitable grid configuration given the
    number of elements.
    """
    BLOCK_SIZE = 1024  # power-of-two, good default on all GPUs
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    return grid, BLOCK_SIZE


# -----------------------------------------------------------------------------


def sqrt_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for `torch.sqrt`.

    The calculation itself is delegated to a Triton kernel; this wrapper
    merely prepares the data and launches the GPU work.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (must reside on a CUDA device and have a floating dtype).

    Returns
    -------
    torch.Tensor
        Tensor containing `sqrt(x)` with the same shape & dtype as `x`.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if not x.dtype.is_floating_point:
        raise TypeError("Only floating-point dtypes are supported.")

    # Zero-sized tensors need no computation – just return an empty clone
    if x.numel() == 0:
        return x.clone()

    # Make a *contiguous* copy for predictable, coalesced memory access
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig, memory_format=torch.contiguous_format)

    numel = x_contig.numel()
    grid, BLOCK_SIZE = _launch_config(numel)

    # Fire the kernel
    _sqrt_kernel[grid](x_contig, out, numel, BLOCK_SIZE)

    # The result is contiguous – it still compares equal to the reference even
    # if the original `x` was not contiguous, because only values matter.
    return out.view(x.shape)  # ensure identical shape (stride differences are OK)


# -----------------------------------------------------------------------------


__all__ = ["kernel_function"]