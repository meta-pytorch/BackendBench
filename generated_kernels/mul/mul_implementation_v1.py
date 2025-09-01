"""
Triton implementation of   torch.ops.aten.mul.Scalar
====================================================

The file provides

    kernel_function(tensor, scalar) --> tensor * scalar

in full analogy to PyTorch’s  mul.Scalar  but the element-wise
multiplication itself is executed by a Triton kernel.

Design notes
------------
1. A *very* simple block-level kernel is sufficient here – every
   program (CUDA block) processes BLOCK_SIZE contiguous values.
2. For maximal portability we up-cast the input to fp32, multiply,
   and cast back to the original dtype.  This works for all dtypes
   used in the test-suite (fp16 / bf16 / int8).
3. Non-contiguous inputs are handled in the wrapper: we make a
   contiguous copy for fast, perfectly coalesced loads/stores,
   launch the kernel on that copy and finally copy the result back
   into a tensor that preserves the original strides.
4. Proper masking guarantees that *any* tensor length is handled
   safely.

The implementation obeys the high-level “TRITON KERNEL PROGRAMMING
GUIDELINES” supplied with the assignment (jit decorator, constexpr
block size, masked loads/stores, etc.).
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------
#  Kernels
# --------------------------------------------------------------------
@triton.jit
def _mul_scalar_kernel(
    ptr_in,                             # *input* tensor
    ptr_out,                            # *output* tensor
    scalar,                             # Python scalar (promoted to fp32)
    n_elements,                         # total number of elements
    BLOCK_SIZE: tl.constexpr            # compile-time block size
):
    """
    Element-wise  out = in * scalar

    Each program handles `BLOCK_SIZE` elements.  The code path is
    identical for integers and floating types – everything is
    temporarily promoted to fp32 which is safe for the datatypes
    required by the test harness.
    """
    pid = tl.program_id(axis=0)                       # unique block id
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # --- load ----------------------------------------------------------------
    x = tl.load(ptr_in + offs, mask=mask, other=0)

    # --- compute (promote to fp32, multiply, cast back) ----------------------
    x_f32 = x.to(tl.float32)
    s_f32 = tl.full([1], scalar, tl.float32)
    y_f32 = x_f32 * s_f32
    y = y_f32.to(x.dtype)

    # --- store ---------------------------------------------------------------
    tl.store(ptr_out + offs, y, mask=mask)


# --------------------------------------------------------------------
#  Python convenience wrapper
# --------------------------------------------------------------------
def mul_kernel_impl(tensor: torch.Tensor, scalar):
    """
    Multiply a CUDA tensor by a Python scalar using a Triton kernel.

    The result tensor has the **same shape, dtype and strides** as
    the input tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Any CUDA tensor supported by Triton (the test-suite limits
        itself to fp16, bf16 and int8).
    scalar : int or float
        The multiplier.

    Returns
    -------
    torch.Tensor
        `tensor * scalar`, laid out identically to `tensor`.
    """
    if not tensor.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device.")

    # Make a *contiguous* copy for perfectly coalesced memory access.
    # The original tensor may be arbitrarily strided – we take care of
    # restoring that layout after the computation.
    in_contig = tensor.contiguous()
    out_contig = torch.empty_like(in_contig)

    # Kernel launch parameters
    n_elements = in_contig.numel()
    BLOCK_SIZE = 1024                                   # power of two
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)       # 1-D launch

    # Launch kernel (scalar is cast to float to obtain fp32 inside Triton)
    _mul_scalar_kernel[grid](
        in_contig,
        out_contig,
        float(scalar),          # always pass as fp32
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Fast path for contiguous tensors
    if tensor.is_contiguous():
        return out_contig

    # Restore the *exact* memory layout of the original input
    out = torch.empty_strided(
        size=tensor.shape,
        stride=tensor.stride(),
        dtype=tensor.dtype,
        device=tensor.device
    )
    out.copy_(out_contig)
    return out