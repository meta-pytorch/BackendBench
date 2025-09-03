# kernel.py
# ----------------------------------------------------------------------
# In-place element-wise division implemented with Triton
#
#   out = input_tensor.div_(divisor)
#
# * The operation is performed **in-place** on `input_tensor`.
# * `divisor` can be a scalar (Python number) or another tensor.
# * Tensor divisors are broadcast-expanded on the host side and passed
#   as a contiguous buffer to the Triton kernel.
# * The kernel works for every floating dtype supported by Triton /
#   PyTorch (this test-suite only uses fp16 & bf16, but fp32 etc. work
#   as well).
#
# ----------------------------------------------------------------------

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Low-level Triton kernel
# ----------------------------------------------------------------------
@triton.jit
def _div_kernel(
    x_ptr,                               # *mutable* input tensor
    y_ptr,                               # divisor tensor (ignored when scalar)
    scalar_div,                          # scalar divisor
    n_elements,                          # total number of elements
    has_tensor_div: tl.constexpr,        # 0 → scalar, 1 → tensor
    BLOCK_SIZE: tl.constexpr,            # work-group size
):
    """
    Vectorised in-place element-wise division.

    Each kernel instance processes `BLOCK_SIZE` contiguous elements from
    the underlying storage of `x_ptr`.  Boundary checks guarantee safe
    execution when `n_elements` is not a multiple of `BLOCK_SIZE`.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    x = tl.load(x_ptr + offsets, mask=mask)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    if has_tensor_div:
        y = tl.load(y_ptr + offsets, mask=mask)
        res = x / y
    else:
        res = x / scalar_div

    # Cast back to original dtype to guarantee correct store type
    res = res.to(x.dtype)

    # ------------------------------------------------------------------
    # Store (in-place)
    # ------------------------------------------------------------------
    tl.store(x_ptr + offsets, res, mask=mask)


# ----------------------------------------------------------------------
# Public user-facing wrapper
# ----------------------------------------------------------------------
def div__kernel_impl(input_tensor: torch.Tensor, divisor):
    """
    High-level wrapper that mimics ``torch.Tensor.div_`` semantics but
    executes the computation with a custom Triton kernel.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor that will be **mutated in-place**.  Must reside on CUDA.
    divisor : float | int | torch.Tensor
        Scalar value or tensor that is broadcast-compatible with
        `input_tensor`.

    Returns
    -------
    torch.Tensor
        The *same* tensor instance that was passed in (`input_tensor`)
        after the in-place division, enabling call chaining just like the
        original PyTorch op.
    """
    # ------------------------------------------------------------------
    # Sanity checks & device placement
    # ------------------------------------------------------------------
    if not input_tensor.is_cuda:
        raise ValueError("`input_tensor` must live on a CUDA device.")

    device = input_tensor.device
    has_tensor_div = isinstance(divisor, torch.Tensor)

    # ------------------------------------------------------------------
    # Prepare divisor
    # ------------------------------------------------------------------
    if has_tensor_div:
        # Ensure the divisor sits on the same device
        divisor = divisor.to(device, non_blocking=True)

        # Materialise broadcasting on the host by creating a contiguous
        # expanded copy.  This keeps the device-side kernel simple and
        # guarantees 1-to-1 correspondence between `x` and `y`.
        if divisor.shape != input_tensor.shape:
            divisor_tensor = divisor.expand(input_tensor.shape).contiguous()
        else:
            divisor_tensor = divisor.contiguous()
        scalar_value = 0.0  # dummy (unused)
    else:
        # Scalar path
        scalar_value = float(divisor)
        # Dummy tensor – never read in the scalar path
        divisor_tensor = input_tensor

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024  # good default (power-of-two, warp-friendly)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # ------------------------------------------------------------------
    # Fire the kernel
    # ------------------------------------------------------------------
    _div_kernel[grid](
        input_tensor,               # x_ptr
        divisor_tensor,             # y_ptr (dummy if scalar path)
        scalar_value,               # scalar divisor
        n_elements,
        has_tensor_div=has_tensor_div,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Return the SAME tensor object (in-place semantics)
    return input_tensor