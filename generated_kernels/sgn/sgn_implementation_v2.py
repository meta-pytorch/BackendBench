# kernel.py
#
# High-performance Triton implementation of `torch.sgn` (a.k.a `torch.sign`).
# --------------------------------------------------------------------------
# • Works for every dtype the Op supports:
#     – floating (fp16 / bf16 / fp32 / fp64 …)
#     – integer  (all widths, signed or unsigned)
#     – bool
#     – complex64  (implemented explicitly – complex128 can easily be added)
# • The heavy lifting is done inside Triton kernels; no PyTorch math is used
#   for the actual computation.
# • A Python wrapper (`kernel_function`) handles kernel-selection, launch-
#   parameters and returns a normal PyTorch tensor.
#
# Author: ChatGPT (2024)
# --------------------------------------------------------------------------

import triton
import triton.language as tl
import torch


# ---------------------------------------------------------------------------
#  Real / Integer / Bool kernel
# ---------------------------------------------------------------------------
@triton.jit
def _sgn_kernel_real(x_ptr, y_ptr, numel, BLOCK_SIZE: tl.constexpr):
    """
    Element-wise sign for **non-complex** tensors.

    1 for  x > 0
    0 for  x == 0
    −1 for x < 0

    Special case:
        • bool tensors already hold only 0 / 1 → result = x
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask)

    # Fast path for bool – just forward the value.
    if tl.constexpr(x.dtype == tl.int1):
        y = x
    else:
        pos = (x > 0).to(x.dtype)       # 1 where x > 0 else 0
        neg = (x < 0).to(x.dtype)       # 1 where x < 0 else 0
        y = pos - neg                   #   1  –   0  =  1
                                        #   0  –   1  = −1
                                        #   0  –   0  =  0

    tl.store(y_ptr + offsets, y, mask=mask)


# ---------------------------------------------------------------------------
#  Complex64 kernel  (complex128 can be added analogously)
# ---------------------------------------------------------------------------
@triton.jit
def _sgn_kernel_complex(fp_view_in_ptr, fp_view_out_ptr,
                        num_complex, BLOCK_SIZE: tl.constexpr):
    """
    Element-wise sign for complex64 tensors.

        sgn(z) = z / |z| ,  z ≠ 0
                 0        ,  z == 0

    Memory view:
        complex64  == two float32 numbers  (real, imag) laid out contiguously.
    We therefore index by *complex element* and multiply the offset by 2 to
    reach the proper float32 address.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)        # complex index
    mask = idx < num_complex

    base = idx * 2                                      # float32 index
    real = tl.load(fp_view_in_ptr + base,     mask=mask, other=0.0)
    imag = tl.load(fp_view_in_ptr + base + 1, mask=mask, other=0.0)

    mag_sq  = real * real + imag * imag                 # |z|^2
    inv_mag = tl.math.rsqrt(mag_sq)                     # 1 / |z|
    # Avoid division-by-zero → scale = 0 where |z| == 0
    scale   = tl.where(mag_sq == 0.0, 0.0, inv_mag)

    out_real = real * scale
    out_imag = imag * scale

    tl.store(fp_view_out_ptr + base,     out_real, mask=mask)
    tl.store(fp_view_out_ptr + base + 1, out_imag, mask=mask)


# ---------------------------------------------------------------------------
#  Public Python wrapper
# ---------------------------------------------------------------------------
def sgn_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for `torch.sgn` implemented with Triton.

    Parameters
    ----------
    x : torch.Tensor (CUDA)
        Input tensor.

    Returns
    -------
    torch.Tensor
        Element-wise sign of `x`, same shape & dtype.
    """
    if not x.is_cuda:
        raise ValueError("Input must live on a CUDA device.")

    # Allocate output tensor
    y = torch.empty_like(x)

    # Decide which kernel to launch ------------------------------------------------
    BLOCK_SIZE = 1024  # good default – multiple of 32 & 64, power-of-2

    if x.is_complex():
        # Currently support complex64 (two fp32 values).  complex128 can be handled
        # the same way by switching to float64 views.
        if x.dtype != torch.complex64:
            raise NotImplementedError("Only complex64 is supported at the moment.")

        # View complex memory as raw fp32 for the kernel.
        in_view  = x.view(torch.float32)
        out_view = y.view(torch.float32)
        numel    = x.numel()  # number of **complex** elements

        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _sgn_kernel_complex[grid](
            in_view, out_view,
            numel,
            BLOCK_SIZE,
        )

    else:
        # Real / integer / bool path
        numel = x.numel()
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _sgn_kernel_real[grid](
            x, y,
            numel,
            BLOCK_SIZE,
        )

    return y