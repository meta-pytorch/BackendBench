# kernel.py
#
# High-performance Triton implementation of `torch.sin`
# ----------------------------------------------------
# This file exposes a single public function – `kernel_function` – that
# behaves exactly like `torch.sin`.  All heavy numerical work is done
# inside a Triton kernel; the Python wrapper is only responsible for
# preparing the launch grid and post-processing the result.
#
# The implementation follows the “TRITON KERNEL PROGRAMMING GUIDELINES”
# given in the task description.
#
# Author: <your-name>
# ---------------------------------------------------------------------

import triton
import triton.language as tl
import torch


# ---------------------------------------------------------------------
# 1. Triton kernel
# ---------------------------------------------------------------------
@triton.jit
def _sin_kernel(
    x_ptr,                           # *  pointer to input  tensor
    y_ptr,                           # *  pointer to output tensor
    n_elements,                      #    total number of scalar values
    BLOCK_SIZE: tl.constexpr,        #    number of threads per block
):
    """
    Vectorised element-wise sine.

    Each program instance (i.e. CUDA block) handles `BLOCK_SIZE` contiguous
    elements.  Out-of-bounds accesses are masked out explicitly.
    """
    pid = tl.program_id(axis=0)                       # current block id
    block_start = pid * BLOCK_SIZE                    # first elem handled by this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vector of indices
    mask = offsets < n_elements                       # OOB mask

    # ------------------------------------------------------------------
    # LOAD → COMPUTE → STORE   (Guideline 5a “Elementwise” pattern)
    # ------------------------------------------------------------------
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # For better numerical accuracy on FP16 / BF16 we promote to FP32,
    # perform the sine, then cast back to the original dtype.
    y_fp32 = tl.sin(x.to(tl.float32))
    y = y_fp32.to(x.dtype)

    tl.store(y_ptr + offsets, y, mask=mask)


# ---------------------------------------------------------------------
# 2. Public wrapper
# ---------------------------------------------------------------------
def sin_kernel_impl(input_tensor: torch.Tensor) -> torch.Tensor:  # noqa: N802
    """
    Drop-in replacement for `torch.sin` backed by Triton.

    Parameters
    ----------
    input_tensor : torch.Tensor
        CUDA tensor of dtype float16 / bfloat16 / float32 / float64.
        (float64 will be down-cast to float32 for the computation and
        then up-cast again; this keeps the interface intact while still
        using fast 32-bit math in the kernel.)

    Returns
    -------
    torch.Tensor
        Output tensor with `sin(input_tensor)` element-wise, same shape,
        dtype and device as the input.
    """
    # --------------------------------------------------------------
    # Basic sanity checks
    # --------------------------------------------------------------
    if not input_tensor.is_cuda:
        raise ValueError("Triton kernel only works on CUDA tensors.")
    if input_tensor.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise TypeError(
            f"dtype {input_tensor.dtype} is not supported by this kernel."
        )

    # Handle empty tensors up-front
    numel = input_tensor.numel()
    if numel == 0:
        return input_tensor.clone()

    # --------------------------------------------------------------
    # Kernel launch parameters
    # --------------------------------------------------------------
    BLOCK_SIZE = 1024  # power of two  (Guideline 4)
    grid = (triton.cdiv(numel, BLOCK_SIZE),)  # 1-D grid  (Guideline 3)

    # --------------------------------------------------------------
    # Memory preparation
    # --------------------------------------------------------------
    # The Triton kernel works with contiguous memory for maximum
    # bandwidth.  A non-contiguous view is copied into a contiguous
    # buffer, processed, and the result is copied back respecting
    # the original strides.  This keeps the interface transparent
    # and still guarantees correctness.
    #
    # (Copying is allowed in the wrapper; the computation itself
    #  *must* happen inside the Triton kernel – see task rules.)
    x_contig = input_tensor.contiguous()
    y_contig = torch.empty_like(x_contig)

    # --------------------------------------------------------------
    # Launch the Triton kernel
    # --------------------------------------------------------------
    _sin_kernel[grid](
        x_contig,                       # x_ptr
        y_contig,                       # y_ptr
        numel,                          # n_elements
        BLOCK_SIZE=BLOCK_SIZE,          # constexpr
    )

    # --------------------------------------------------------------
    # Re-shape / re-stride result to match the original tensor
    # --------------------------------------------------------------
    if input_tensor.is_contiguous():
        return y_contig  # fast-path
    else:
        # Allocate an output tensor with the same *strides* as the
        # original view and copy the contiguous data over.
        out = torch.empty_like(input_tensor)
        out.copy_(y_contig.view_as(out))   # device-to-device copy
        return out


# ---------------------------------------------------------------------
# 3. Module self-test (optional – runs only when executed directly)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test mirroring the official test-driver logic
    from math import pi

    torch.manual_seed(0)
    x = torch.linspace(-2 * pi, 2 * pi, steps=1024, device="cuda", dtype=torch.float32)
    ref = torch.sin(x)
    out = kernel_function(x)

    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6)
    print("✅  quick self-test passed")