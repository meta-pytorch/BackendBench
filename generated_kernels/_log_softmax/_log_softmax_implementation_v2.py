# kernel.py
#
# Triton implementation of
#     aten._log_softmax.default
# for 2-D tensors (float16 / bfloat16 / float32).  All mathematical work
# is performed inside a Triton kernel – **no** PyTorch math ops are used
# in the critical path.
#
# Public entry-point  :  kernel_function(x, dim, half_to_float)
#
# ----------------------------------------------------------------------
# Implementation notes
# ----------------------------------------------------------------------
# • One Triton *program* = one logical “row” to be reduced.  When
#   dim==1 this is the true tensor row; when dim==0 we just reinterpret
#   memory so that each program walks down a physical column.
# • The computation is split in the textbook three-pass scheme:
#       (1) max reduction          – avoid overflow
#       (2) Σ exp(x − max)         – still in fp32
#       (3) final transform / store
# • All intermediate math uses fp32 for accuracy.  The output dtype is
#   chosen according to PyTorch’s rules:
#       – same as input, **except**  fp16 + half_to_float=True → fp32
# • Boundary masking is handled with ‑inf sentinels so that ignored
#   elements do not pollute the reductions (important for short rows).
#
# ----------------------------------------------------------------------

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
@triton.jit
def _log_softmax_kernel(
        x_ptr,                       # *const T     – input  base-ptr
        o_ptr,                       # *T_out       – output base-ptr
        ROWS: tl.constexpr,          # number of logical rows
        COLS: tl.constexpr,          # length of each row
        STRIDE_ROW: tl.constexpr,    # stride between rows    (elements)
        STRIDE_COL: tl.constexpr,    # stride between columns (elements)
        BLOCK_SIZE: tl.constexpr     # elements processed per loop
):
    """
    Each program handles one logical row (pid).  Inside the row we iterate
    with a vector of size BLOCK_SIZE until all COLS elements are processed.
    """

    pid = tl.program_id(axis=0)
    if pid >= ROWS:
        return

    # Base element offset of the row start
    row_offset = pid * STRIDE_ROW
    offs       = tl.arange(0, BLOCK_SIZE)

    # --------------------------------------------------------------
    # (1) Row-wise maximum
    # --------------------------------------------------------------
    neg_inf = -float("inf")
    row_max = tl.full([], neg_inf, tl.float32)

    for start in tl.range(0, COLS, BLOCK_SIZE):
        idx   = start + offs
        mask  = idx < COLS
        ptrs  = x_ptr + row_offset + idx * STRIDE_COL
        x     = tl.load(ptrs, mask=mask, other=neg_inf).to(tl.float32)
        cur_m = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, cur_m)

    # --------------------------------------------------------------
    # (2) Row-wise Σ exp(x − max)
    # --------------------------------------------------------------
    row_sum_exp = tl.zeros([], dtype=tl.float32)

    for start in tl.range(0, COLS, BLOCK_SIZE):
        idx   = start + offs
        mask  = idx < COLS
        ptrs  = x_ptr + row_offset + idx * STRIDE_COL
        x     = tl.load(ptrs, mask=mask, other=neg_inf).to(tl.float32)
        row_sum_exp += tl.sum(tl.exp(x - row_max), axis=0)

    log_row_sum_exp = tl.log(row_sum_exp)

    # --------------------------------------------------------------
    # (3) Final output
    # --------------------------------------------------------------
    for start in tl.range(0, COLS, BLOCK_SIZE):
        idx      = start + offs
        mask     = idx < COLS
        in_ptrs  = x_ptr + row_offset + idx * STRIDE_COL
        out_ptrs = o_ptr + row_offset + idx * STRIDE_COL

        x = tl.load(in_ptrs, mask=mask).to(tl.float32)
        y = x - row_max - log_row_sum_exp

        # Cast to the *output* element type before storing
        tl.store(out_ptrs, y.to(o_ptr.dtype.element_ty), mask=mask)


# ----------------------------------------------------------------------
# Python wrapper
# ----------------------------------------------------------------------
def _log_softmax_kernel_impl(x: torch.Tensor,
                    dim: int,
                    half_to_float: bool = False) -> torch.Tensor:
    """
    Parameters
    ----------
    x   : 2-D CUDA tensor (fp16 / bf16 / fp32)
    dim : reduction dimension (0 or 1, negative indices allowed)
    half_to_float : follow PyTorch’s behaviour
                    (fp16 input + True → fp32 output)

    Returns
    -------
    A tensor with the same shape as `x` and with the correct dtype.
    """

    # --------------------------- sanity --------------------------------
    if not x.is_cuda:
        raise RuntimeError("Input tensor must live on CUDA")
    if x.dim() != 2:
        raise RuntimeError("Only 2-D tensors are supported")

    # Canonicalise dim to {0, 1}
    dim = dim % 2

    # Decide output dtype according to PyTorch semantics
    if x.dtype == torch.float16 and half_to_float:
        out_dtype = torch.float32
    else:
        out_dtype = x.dtype

    # ------------------------------------------------------------------
    #     Build logical ROW/COL view + element-strides
    # ------------------------------------------------------------------
    if dim == 1:                       # reduce over last dimension
        ROWS, COLS = x.shape
        stride_row = x.stride(0)
        stride_col = x.stride(1)
    else:                              # reduce over first dimension
        ROWS, COLS = x.shape[1], x.shape[0]
        stride_row = x.stride(1)
        stride_col = x.stride(0)

    # Allocate output
    out = torch.empty_like(x, dtype=out_dtype)

    # Kernel launch configuration
    BLOCK_SIZE = 1024
    grid = (ROWS,)                     # 1-D grid – one program per row

    _log_softmax_kernel[grid](
        x, out,
        ROWS, COLS,
        stride_row, stride_col,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )

    return out