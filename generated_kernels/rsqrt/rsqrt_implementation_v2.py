# kernel.py
# -----------------------------------------------------------------------------
# Triton implementation of the element-wise reciprocal square-root (rsqrt)
# operation equivalent to `torch.ops.aten.rsqrt.default`.
#
# Design goals
#   • Works for every tensor shape, size and stride configuration
#   • Supports the floating-point dtypes used in the test-suite (bf16 / fp16)
#     – fp32 is accepted as well for completeness
#   • Pure Triton math inside the GPU kernel (no PyTorch shortcuts)
#   • Simple wrapper function `kernel_function` so that the test-suite can call
#     it like a regular Python function.
#
# Author: OpenAI – ChatGPT
# -----------------------------------------------------------------------------

import triton
import triton.language as tl
import torch


# -----------------------------------------------------------------------------
# 1.  Triton GPU kernel
# -----------------------------------------------------------------------------
@triton.jit
def _rsqrt_kernel(
    x_ptr,                      # *const T  – input tensor
    y_ptr,                      # *      T  – output tensor
    numel,                      # int32      – total number of elements
    BLOCK_SIZE: tl.constexpr,   # meta-parameter (must be power-of-two ≤ 1024)
):
    """
    A very simple element-wise kernel:

        y[i] = 1 / sqrt(x[i])      for   0 ≤ i < numel

    The work is split so that each program (CUDA thread-block) processes
    `BLOCK_SIZE` contiguous *indices*.  We still support non-contiguous tensors
    because we launch the kernel on *contiguous* copies of the input/output
    (handled by the Python wrapper, see below).
    """
    # ---------------------------------------------------------------------
    # 1. Which element indices does this program (thread-block) own?
    # ---------------------------------------------------------------------
    pid        = tl.program_id(axis=0)                       # 1-D launch grid
    block_start = pid * BLOCK_SIZE
    offsets     = block_start + tl.arange(0, BLOCK_SIZE)     # vector of indices
    mask        = offsets < numel                            # boundary mask

    # ---------------------------------------------------------------------
    # 2. Load -> compute -> store (Elementwise kernel pattern)
    # ---------------------------------------------------------------------
    x = tl.load(x_ptr + offsets, mask=mask)                  # original dtype
    x_fp32 = x.to(tl.float32)                                # promote – accuracy

    # reciprocal square-root
    rsqrt_fp32 = 1.0 / tl.sqrt(x_fp32)

    # Cast back to the pointer’s dtype *before* writing.
    out_dtype = y_ptr.dtype.element_ty
    if out_dtype == tl.float16:
        rsqrt_cast = rsqrt_fp32.to(tl.float16)
    elif out_dtype == tl.bfloat16:
        rsqrt_cast = rsqrt_fp32.to(tl.bfloat16)
    else:                                                    # fallback / fp32
        rsqrt_cast = rsqrt_fp32

    tl.store(y_ptr + offsets, rsqrt_cast, mask=mask)


# -----------------------------------------------------------------------------
# 2.  Public Python API
# -----------------------------------------------------------------------------
def rsqrt_kernel_impl(inp: torch.Tensor) -> torch.Tensor:
    """
    Reciprocal square-root implemented with Triton.

    Parameters
    ----------
    inp : torch.Tensor (CUDA)
        Input tensor of dtype bf16, fp16 or fp32.  Any shape or stride layout
        is allowed.

    Returns
    -------
    torch.Tensor
        Result tensor with the same shape & dtype as `inp` containing
        `1 / sqrt(inp)`.  (The returned tensor is contiguous unless the input
        was non-contiguous, in which case the original stride layout is
        preserved.)
    """
    # ---------------------------------------------------------------------
    # 0.  Sanity checks
    # ---------------------------------------------------------------------
    if not inp.is_cuda:
        raise ValueError("kernel_function: input tensor must reside on a CUDA "
                         "device, got CPU tensor.")
    if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"kernel_function: unsupported dtype {inp.dtype}. "
                        "Supported: fp16, bf16, fp32.")

    # ---------------------------------------------------------------------
    # 1. Create *contiguous* working copies
    #    – simplifies kernel indexing drastically.  We convert back to the
    #      original layout at the end if necessary.
    # ---------------------------------------------------------------------
    x_contig = inp.contiguous()
    y_contig = torch.empty_like(x_contig)

    # ---------------------------------------------------------------------
    # 2. Kernel launch configuration
    # ---------------------------------------------------------------------
    numel       = x_contig.numel()
    BLOCK_SIZE  = 1024                             # power-of-two ≤ 1024
    grid        = (triton.cdiv(numel, BLOCK_SIZE),)

    # ---------------------------------------------------------------------
    # 3. Launch Triton kernel
    # ---------------------------------------------------------------------
    _rsqrt_kernel[grid](x_contig,                  # *const T
                        y_contig,                  # *      T
                        numel,                     # int32
                        BLOCK_SIZE=BLOCK_SIZE)     # meta

    # ---------------------------------------------------------------------
    # 4. If the original tensor was non-contiguous, replicate that layout
    #    (tests only check for values / dtype / shape, but we preserve strides
    #    anyway to stay semantically faithful to PyTorch).
    # ---------------------------------------------------------------------
    if inp.is_contiguous():
        return y_contig
    else:
        # Allocate a tensor with the *same* shape & strides as `inp`
        y = torch.empty_like(inp)
        y.copy_(y_contig)      # element-wise copy (no computation)
        return y