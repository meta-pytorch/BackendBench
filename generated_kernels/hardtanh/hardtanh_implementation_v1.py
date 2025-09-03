# kernel.py
#
# High-performance HardTanh implementation backed by a Triton kernel.
#
# The public entry-point is `kernel_function`, which has the same calling
# convention as  `torch.ops.aten.hardtanh.default`:
#
#     out = kernel_function(inp, min_val, max_val)
#
# The core computation (clamp to the closed interval [min_val, max_val])
# is performed entirely in Triton – no PyTorch math ops are used inside
# the kernel itself.  The wrapper only handles argument checking, memory
# allocation and kernel launch.

import torch
import triton
import triton.language as tl


###############################################################################
#                               TRITON KERNEL                                 #
###############################################################################
@triton.jit
def _hardtanh_kernel(x_ptr,                             # * ptr to input
                     y_ptr,                             # * ptr to output
                     numel,                             # total number of elements
                     min_val, max_val,                  # scalar clip bounds
                     BLOCK_SIZE: tl.constexpr):         # how many elements per block
    """
    A very small, purely element-wise kernel:
        y[i] = clamp(x[i], min_val, max_val)

    Each program instance (i.e. CUDA block) processes `BLOCK_SIZE`
    consecutive elements.  The last block is masked to avoid
    out-of-bounds accesses.
    """
    pid = tl.program_id(axis=0)            # unique block id
    block_start = pid * BLOCK_SIZE         # first element this program handles
    offsets = block_start + tl.arange(0, BLOCK_SIZE)    # vector of indices
    mask = offsets < numel                 # mask for the ragged last block

    # --------------------------------------------------------------------- #
    #                               LOAD                                    #
    # --------------------------------------------------------------------- #
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # --------------------------------------------------------------------- #
    #                              COMPUTE                                  #
    # --------------------------------------------------------------------- #
    # Perform the clamp in FP32 for a bit more accuracy, then cast back
    # to the original dtype (BF16 / FP16 / FP32, …).
    x_fp32 = x.to(tl.float32)

    # First apply the lower bound, then the upper bound.
    x_fp32 = tl.where(x_fp32 < min_val, min_val, x_fp32)
    x_fp32 = tl.where(x_fp32 > max_val, max_val, x_fp32)

    y = x_fp32.to(x.dtype)

    # --------------------------------------------------------------------- #
    #                               STORE                                   #
    # --------------------------------------------------------------------- #
    tl.store(y_ptr + offsets, y, mask=mask)


###############################################################################
#                           PYTHON WRAPPER API                                #
###############################################################################
def hardtanh_kernel_impl(inp: torch.Tensor,
                    min_val: float,
                    max_val: float) -> torch.Tensor:
    """
    Apply the HardTanh activation to `inp` using a Triton kernel.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor located on a CUDA device.  Supported dtypes: bfloat16,
        float16, float32.
    min_val, max_val : float
        Hard lower / upper clipping thresholds.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape, dtype and device as `inp` containing
        the clamped results (memory layout may be different).
    """
    # ---------------------------  Sanity checks  --------------------------- #
    if not inp.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device")
    if min_val > max_val:
        raise ValueError("`min_val` must be <= `max_val`")
    if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Supported dtypes are: float16, bfloat16 and float32")

    # For simplicity the kernel operates on contiguous buffers.  This has
    # no impact on numerical correctness and mirrors the behaviour of
    # PyTorch’s out-of-place HardTanh (which also returns a contiguous
    # tensor).
    x_contig = inp.contiguous()
    out = torch.empty_like(x_contig)

    # -------------------------  Kernel launch  ---------------------------- #
    numel = x_contig.numel()
    BLOCK_SIZE = 1024                       # power-of-two → better coalescing
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    _hardtanh_kernel[grid](
        x_contig, out,
        numel,
        float(min_val),                     # Triton expects python floats here
        float(max_val),
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape to the original logical shape;  the underlying storage is
    # already correct.
    return out.view_as(inp)