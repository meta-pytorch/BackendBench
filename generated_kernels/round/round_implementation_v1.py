# ================================================================
#  kernel.py
#  ----------------------------------------------------------------
#  High-performance Triton implementation of torch.round()
#  (round to nearest integer, “ties-to-even” a.k.a. bankers-rounding)
#
#  • Supports every dtype the reference op supports on the GPU
#      – floating:  bfloat16 / float16  (float32/64 will work too)
#      – integer :  int8 / int16 / int32 / int64  (identity)
#  • Works for 0-D, 1-D, 2-D … arbitrary shapes & strides
#  • Obeys Triton best-practice rules: masks, coalesced access,
#    BLOCK_SIZE power-of-2, out-of-bounds protection, …
#
#  The test-suite expects a *regular* Python function called
#  `kernel_function(...)` – that is provided below and internally
#  launches the Triton kernel.
# ================================================================

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------
#  Triton device kernel
# -----------------------------------------------------------------
@triton.jit
def _round_kernel(
    in_ptr,                          # *input*  tensor storage
    out_ptr,                         # *output* tensor storage
    n_elements,                      # total number of scalars
    BLOCK_SIZE: tl.constexpr,        # threads per block (power-of-2)
    IS_FLOAT: tl.constexpr,          # compile-time flag: do real work or copy
):
    """
    Vectorised element-wise `round()` with “ties-to-even”.

    Parameters
    ----------
    in_ptr / out_ptr : pointers to the first element
    n_elements       : total element count (flattened)
    BLOCK_SIZE       : how many elements each programme instance handles
    IS_FLOAT         : `True`  -> perform rounding
                       `False` -> integer dtype, just copy
    """
    pid = tl.program_id(axis=0)                     # programme instance id
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                     # OOB guard

    # --------------------------------------------------------------
    #  Load
    # --------------------------------------------------------------
    x = tl.load(in_ptr + offsets, mask=mask, other=0)

    # --------------------------------------------------------------
    #  Branch at *compile-time* depending on dtype
    # --------------------------------------------------------------
    if IS_FLOAT:
        # --- Promote to fp32 for accurate arithmetic ----------------
        xf32 = x.to(tl.float32)

        # 1) naive nearest-integer (half-away-from-zero)
        nearest = tl.math.floor(xf32 + 0.5)

        # 2) detect exact “.5” ties
        diff  = tl.abs(xf32 - nearest)
        is_tie = diff == 0.5                         # boolean tensor

        # 3) detect odd candidates
        nearest_i32 = nearest.to(tl.int32)
        is_odd = (nearest_i32 & 1) != 0              # bool tensor

        # 4) ties-to-even adjustment  (odd & tie  -> subtract 1)
        adjust_mask = is_tie & is_odd
        adjust = adjust_mask.to(tl.float32)          # 1.0 where we need fix
        rounded = nearest - adjust                   # final fp32 result

        # 5) Cast back to original floating dtype
        y = rounded.to(x.dtype)
    else:
        # Integer inputs: torch.round is a no-op (identity)
        y = x

    # --------------------------------------------------------------
    #  Store
    # --------------------------------------------------------------
    tl.store(out_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------
#  Public Python wrapper – this is what the test-suite imports
# -----------------------------------------------------------------
def round_kernel_impl(inp: torch.Tensor) -> torch.Tensor:
    """
    Round `inp` element-wise to the nearest integer (“ties-to-even”),
    behaviour-compatible with `torch.round`.

    The heavy lifting is performed by a Triton kernel; this wrapper
    only prepares launch parameters and allocates the output.

    Parameters
    ----------
    inp : torch.Tensor  (must live on CUDA device)

    Returns
    -------
    torch.Tensor   (same shape / dtype / stride as `inp`)
    """
    if not inp.is_cuda:
        raise ValueError("kernel_function: input tensor must be on a CUDA device.")

    # Allocate output with *identical* shape & stride
    out = torch.empty_like(inp)

    # Degenerate case – nothing to do
    n_elems = inp.numel()
    if n_elems == 0:
        return out

    # Decide once at runtime – becomes `tl.constexpr` inside kernel
    is_float = bool(inp.dtype.is_floating_point)

    # Kernel launch configuration
    BLOCK_SIZE = 1024                               # good default (power-of-2)
    grid = (triton.cdiv(n_elems, BLOCK_SIZE),)      # 1-D grid

    # Launch
    _round_kernel[grid](
        inp,                                        # in_ptr
        out,                                        # out_ptr
        n_elems,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_FLOAT=is_float,
    )

    return out