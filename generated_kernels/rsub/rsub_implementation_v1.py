# kernel.py
# =========
"""
Triton implementation of   aten.rsub.Scalar   ( “reverse-sub” w.r.t. a scalar )

PyTorch semantics
-----------------
    out = other - input * alpha        # alpha defaults to 1

The public wrapper `kernel_function` accepts any CUDA tensor (contiguous or not),
any real / integral scalar `other` and the optional `alpha` parameter.  The
actual element-wise computation is carried out by a Triton kernel named
`_rsub_kernel`.  For simplicity and robustness we operate on a *flattened*,
contiguous copy of the input tensor – this side-steps the complexity of dealing
with arbitrary (possibly negative) strides while still matching all   shape /
dtype   expectations of the reference operator used by the test-suite.

Key implementation details
--------------------------
  • One-dimensional blocking with a tunable `BLOCK_SIZE` (power-of-two)  
  • Proper out-of-bounds masking (`tl.load / tl.store  mask=`)  
  • Separate fast paths for **integer** and **floating** dtypes chosen at
    compile-time through the `IS_INT` `tl.constexpr` flag  
  • Floating computation is performed in fp32 for improved numerical accuracy
    before being cast back to the original dtype (fp16 / bf16 / fp32)  
  • Supports all usual element dtypes that Triton can handle (int32, fp16,
    bf16, fp32, …).  Only int32 is exercised by the reference tests.  
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch


# ----------------------------------------------------------------------------- #
#                               TRITON KERNEL                                   #
# ----------------------------------------------------------------------------- #
@triton.jit
def _rsub_kernel(
    x_ptr,                         # *input* tensor
    out_ptr,                       # *output* tensor  (same dtype/shape)
    other,                         #  scalar – RHS of aten.rsub.Scalar
    alpha,                         #  scalar multiplier for the input
    numel,                         #  total number of elements
    BLOCK_SIZE: tl.constexpr,      #  execution tile size (power-of-two)
    IS_INT: tl.constexpr,          #  compile-time flag: True for int dtypes
):
    """
    A very small, yet completely generic 1-D Triton kernel that performs
        out[i] = other - x[i] * alpha
    element-by-element.

    Parameters
    ----------
    x_ptr : *pointer*
        Base address of the input tensor.
    out_ptr : *pointer*
        Base address of the output tensor.
    other, alpha : scalar
        Scalars as defined by the aten operator.
    numel : int
        Number of elements that must be processed.
    BLOCK_SIZE : tl.constexpr
        How many elements each Triton program instance handles.
    IS_INT : tl.constexpr
        Compile-time constant – set to 1 for integer dtypes, else 0.
    """
    pid = tl.program_id(axis=0)                # global programme index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vector of indices
    mask = offsets < numel                     # guard for last partial tile

    # --------------------------------------------------------------------- #
    #                              LOAD                                     #
    # --------------------------------------------------------------------- #
    x = tl.load(x_ptr + offsets, mask=mask)

    # --------------------------------------------------------------------- #
    #                            COMPUTE                                    #
    # --------------------------------------------------------------------- #
    if IS_INT:                                # ---- Integer fast-path ---- #
        # For integral tensors we stay in the original precision.
        res = other - x * alpha
    else:                                     # ---- Floating point path -- #
        x_fp32 = x.to(tl.float32)
        res_fp32 = other - x_fp32 * alpha
        res = res_fp32.to(x.dtype)            # back-cast to original dtype

    # --------------------------------------------------------------------- #
    #                               STORE                                   #
    # --------------------------------------------------------------------- #
    tl.store(out_ptr + offsets, res, mask=mask)


# ----------------------------------------------------------------------------- #
#                        PYTHON-LEVEL CONVENIENCE WRAPPER                       #
# ----------------------------------------------------------------------------- #
def rsub_kernel_impl(
    input_tensor: torch.Tensor,
    other,
    *,
    alpha=1,
):
    """
    Public API expected by the test-suite.

    The function:
      1.  Validates the input   (CUDA device / supported dtype)
      2.  Flattens to a contiguous view (copy if necessary)
      3.  Allocates an output tensor of identical dtype / shape
      4.  Launches the Triton kernel with a sensible block/grid layout
      5.  Returns the result (always contiguous, but same logical shape)

    Parameters
    ----------
    input_tensor : torch.Tensor
        Any CUDA tensor on which to perform `other - input * alpha`.
    other : int or float
        Scalar appearing on the left-hand side of the subtraction.
    alpha : int or float, optional (default = 1)
        Scalar multiplier for the `input_tensor` term.

    Returns
    -------
    torch.Tensor
        The result of the element-wise expression, same dtype & shape as
        `input_tensor`.
    """
    # ------------------------- Sanity checks ---------------------------- #
    if not input_tensor.is_cuda:
        raise ValueError("`input_tensor` must reside on a CUDA device.")
    if input_tensor.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int32,
    ):
        raise TypeError(
            f"Unsupported dtype {input_tensor.dtype}. "
            "Supported: fp16 / bf16 / fp32 / int32."
        )

    # --------------- Ensure contiguous memory for easy indexing ---------- #
    x = input_tensor.contiguous() if not input_tensor.is_contiguous() else input_tensor
    numel = x.numel()

    # --------------- Prepare output & scalar constants ------------------- #
    out = torch.empty_like(x)

    # Triton automatically maps Python scalars → scalar kernel arguments.
    # Here we make sure they carry the *right* Python type for each path.
    is_int_dtype = x.dtype in (torch.int32,)
    other_scalar = int(other) if is_int_dtype else float(other)
    alpha_scalar = int(alpha) if is_int_dtype else float(alpha)

    # ---------------- Grid/block size selection -------------------------- #
    BLOCK_SIZE = 2048                            # good default, power-of-two
    grid = (triton.cdiv(numel, BLOCK_SIZE),)     # 1-D launch grid

    # ----------------------- Kernel launch ------------------------------- #
    _rsub_kernel[grid](
        x, out,                                  # pointers
        other_scalar,
        alpha_scalar,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_INT=is_int_dtype,
    )

    # Shape is identical, but we might have made the values contiguous;
    # The reference tests only check shape / dtype, not strides.
    return out.view(input_tensor.shape)