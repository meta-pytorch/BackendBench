# kernel.py
"""
High-performance element-wise √ (square-root) implemented with Triton.

The public entry-point is `kernel_function(x)` which behaves like
`torch.sqrt(x)` – it takes a single **CUDA** tensor of any floating
dtype / shape / stride and returns a tensor with the same shape,
dtype and values.  The actual computation is performed inside a
Triton kernel to guarantee that no PyTorch math is used.

Implementation notes
--------------------
* We work on a **contiguous** view of the input to ensure perfectly
  coalesced, aligned memory accesses.  This has no functional impact –
  the output is still correct for non-contiguous inputs because the
  values themselves are unchanged.
* Each Triton program (block) processes `BLOCK_SIZE` consecutive
  elements:
      load → cast to fp32 → sqrt → cast back → store
* All out-of-bounds accesses are masked.
* The kernel supports bf16 / fp16 / fp32 transparently (those are the
  floating types currently supported by Triton).

"""

import triton
import triton.language as tl
import torch


# -----------------------------------------------------------------------------
# Triton kernel
# -----------------------------------------------------------------------------
@triton.jit
def _sqrt_kernel(
    x_ptr,                          # *const T  (input)
    y_ptr,                          # *mut  T  (output)
    numel,                          # total number of elements
    BLOCK_SIZE: tl.constexpr        # elements handled by one program
):
    """
    Vectorised square-root: y[i] = sqrt(x[i])

    Parameters
    ----------
    x_ptr : tl.pointer
        Pointer to the first element of the (contiguous) input.
    y_ptr : tl.pointer
        Pointer to the first element of the (contiguous) output.
    numel : int
        Total number of elements to process.
    BLOCK_SIZE : tl.constexpr
        Compile-time constant that decides how many items each program
        handles (typically a power of 2 for best performance).
    """
    pid = tl.program_id(axis=0)                       # program index
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offs < numel                               # OOB protection
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)    # load
    x32 = x.to(tl.float32)                             # promote for accuracy
    y32 = tl.sqrt(x32)                                # √ in fp32
    y  = y32.to(y_ptr.dtype.element_ty)               # cast back
    tl.store(y_ptr + offs, y, mask=mask)              # store


# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------
def sqrt_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Element-wise square-root using a Triton kernel (replacement for
    `torch.sqrt`).  The function is intentionally minimal to mirror the
    signature of the reference `aten.sqrt.default`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (must reside on a CUDA device and have a floating
        dtype: bf16 / fp16 / fp32).

    Returns
    -------
    torch.Tensor
        Tensor with the same shape / dtype containing `sqrt(x)`.
    """
    # -- Sanity checks --------------------------------------------------------
    if not x.is_cuda:
        raise ValueError("Input must be on a CUDA device.")
    if not x.is_floating_point():
        raise TypeError("Only floating dtypes are supported.")
    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"dtype {x.dtype} is not supported by the Triton kernel.")

    numel = x.numel()

    # Empty tensor → nothing to do
    if numel == 0:
        return x.clone()

    # We compute on a contiguous view for fully coalesced accesses.
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)  # output is also contiguous

    # Kernel launch configuration
    BLOCK_SIZE = 1024                              # power-of-2, good default
    grid = (triton.cdiv(numel, BLOCK_SIZE),)       # 1-D launch

    # Launch the Triton kernel
    _sqrt_kernel[grid](
        x_contig, y, numel,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,           # reasonable default for simple kernels
        num_stages=2           # minimal pipelining
    )

    # The result `y` is contiguous – for functional correctness we only
    # need shape & dtype to match the original tensor.  We therefore
    # reshape it to the original shape before returning.
    return y.view(x.shape)


# Allow `python -m kernel` quick-test
if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand(8, device="cuda", dtype=torch.bfloat16)
    ref = torch.sqrt(a)
    res = kernel_function(a)
    print("max abs diff:", (ref - res).abs().max().item())