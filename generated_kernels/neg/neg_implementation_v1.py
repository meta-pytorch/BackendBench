###############################################################################
# kernel.py – Triton implementation of `aten.neg.default`
#
# This file provides a drop-in replacement for `torch.neg` that is entirely
# computed on the GPU by a Triton kernel.  It supports:
#   • floating-point dtypes  : fp16 / bf16 / fp32 / fp64
#   • signed integer dtypes  : int8 / int16 / int32 / int64
#   • complex dtypes         : complex64 / complex128   (handled as two floats)
#
# The public API is the Python function `kernel_function`, which can be called
# exactly like `torch.neg`.  All launch details are hidden inside the wrapper.
###############################################################################

import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------------- #
#  Helper: PyTorch ↔ Triton dtype conversion                                    #
# ----------------------------------------------------------------------------- #
_TORCH2TRITON = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
}

# Complex numbers are handled via their real component dtype
_COMPLEX_BASE_DTYPE = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


# ----------------------------------------------------------------------------- #
#  Triton kernel: element-wise negation                                         #
# ----------------------------------------------------------------------------- #
@triton.jit
def _neg_kernel(ptr_in, ptr_out, numel, BLOCK_SIZE: tl.constexpr,
                DTYPE: tl.constexpr):
    """
    Parameters
    ----------
    ptr_in  : *void       – pointer to the input  tensor buffer
    ptr_out : *void       – pointer to the output tensor buffer
    numel   : int32       – number of **scalar** elements to process
    BLOCK_SIZE : constexpr – how many elements each program instance handles
    DTYPE   : constexpr   – Triton dtype of the *scalar* elements
    """
    # Program-id along the 1-D grid
    pid = tl.program_id(axis=0)

    # Compute the element indices this program will handle
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel                     # boundary protection

    # Load, negate, store --------------------------------------------------- #
    x = tl.load(ptr_in + offs, mask=mask, other=0)
    y = -x
    tl.store(ptr_out + offs, y, mask=mask)
    # ----------------------------------------------------------------------- #


# ----------------------------------------------------------------------------- #
#  Public wrapper – this is what the test-suite imports and calls               #
# ----------------------------------------------------------------------------- #
def neg_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Element-wise negation implemented with Triton.

    This function is 1-for-1 compatible with `torch.neg` (a.k.a.
    `aten.neg.default`).  The computation itself is performed by the Triton
    kernel `_neg_kernel`; PyTorch is used only for tensor book-keeping.

    Parameters
    ----------
    x : torch.Tensor (CUDA)
        Input tensor of any shape / stride / dtype supported by `torch.neg`.

    Returns
    -------
    torch.Tensor
        A tensor with identical shape and dtype where every element is `-x`.
    """
    if not x.is_cuda:
        raise ValueError("`kernel_function` requires the input tensor to live "
                         "on a CUDA device.")

    # ------------------------------------------------------------------ #
    # Fast exit for degenerate (empty) tensors                            #
    # ------------------------------------------------------------------ #
    if x.numel() == 0:
        return x.clone()

    orig_dtype = x.dtype
    is_complex = orig_dtype.is_complex

    # Resolve the *scalar* base dtype (complex -> underlying float)
    base_torch_dtype = _COMPLEX_BASE_DTYPE.get(orig_dtype, orig_dtype)
    if base_torch_dtype not in _TORCH2TRITON:
        raise TypeError(f"Unsupported dtype for neg kernel: {orig_dtype}")

    triton_dtype = _TORCH2TRITON[base_torch_dtype]

    # ------------------------------------------------------------------ #
    # Create contiguous buffers – greatly simplifies addressing logic     #
    # ------------------------------------------------------------------ #
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    # Flatten the view to operate on raw scalars.
    # For complex tensors we treat them as an array of twice as many floats.
    x_scalar = x_contig.view(base_torch_dtype) if is_complex else x_contig
    out_scalar = out.view(base_torch_dtype) if is_complex else out
    num_scalar_elements = x_scalar.numel()

    # ------------------------------------------------------------------ #
    # Launch configuration                                               #
    # ------------------------------------------------------------------ #
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_scalar_elements, BLOCK_SIZE),)

    _neg_kernel[grid](
        x_scalar,               # ptr_in
        out_scalar,             # ptr_out
        num_scalar_elements,    # total # of *scalar* elements
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=triton_dtype,
        num_warps=4,
    )

    return out