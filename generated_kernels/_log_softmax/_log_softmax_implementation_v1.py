# kernel.py
#
# Triton implementation of aten._log_softmax.default
#
# A drop-in replacement for torch.ops.aten._log_softmax.default that
# satisfies the requirements laid out in the test-harness snippet that
# accompanies this file.  Only Triton is used for the mathematical
# work – PyTorch is restricted to (1) memory allocation and (2) trivial
# tensor re-ordering so that the reduction axis becomes the last,
# contiguous dimension.

import math
from typing import List, Tuple, Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------

def _canonicalize_dim(dim: int, rank: int) -> int:
    """
    Turn a possibly-negative `dim` into its positive counterpart.
    """
    if dim < 0:
        dim += rank
    if not (0 <= dim < rank):
        raise ValueError(f"dim={dim} out of range for rank={rank}")
    return dim


def _torch_to_triton_dtype(dtype: torch.dtype):
    """
    Map a torch.dtype to the corresponding tl.* dtype.
    """
    mapping = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported dtype {dtype}")
    return mapping[dtype]


def _move_reduction_axis_to_last(x: torch.Tensor,
                                 dim: int) -> Tuple[torch.Tensor, Optional[List[int]]]:
    """
    Permute `x` such that `dim` becomes the last dimension.  A new
    contiguous tensor is returned to guarantee that the last dimension
    has stride 1.  The inverse permutation is returned as well so the
    caller can restore the original ordering.
    """
    if dim == x.ndim - 1:
        # Already last dimension – just make sure data is contiguous
        return x.contiguous(), None

    perm: List[int] = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]      # swap
    inv_perm: List[int] = [0] * x.ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    x_perm = x.permute(*perm).contiguous()
    return x_perm, inv_perm


# ---------------------------------------------------------------------
# Triton kernel (log-softmax along the *last* dimension)
# ---------------------------------------------------------------------

@triton.jit
def _log_softmax_lastdim_kernel(
    x_ptr,                              # * ptr to the input
    o_ptr,                              # * ptr to the output
    K,                                  #   size of the last  dim  (runtime)
    BLOCK_SIZE: tl.constexpr,           #   threads per program
    ACC_TYPE: tl.constexpr              #   accumulation dtype (fp16/fp32/…)
):
    """
    Each Triton program handles one *row* (i.e. all `K` elements along the
    last / reduction dimension).

    Algorithm (three-pass, numerically stable):
        1) row_max  = max_i   x[i]
        2) row_sum  = sum_i   exp(x[i] - row_max)
           log_sum  = log(row_sum)
        3) out[i]   = x[i] - row_max - log_sum
    """
    pid = tl.program_id(0)                      # row index
    row_start = pid * K                         # offset of the row

    # --------------------------------------------------
    # Pass 1 – compute the per-row maximum
    # --------------------------------------------------
    offs = tl.arange(0, BLOCK_SIZE)             # [0, 1, …, BLOCK_SIZE-1]
    cur_max = -float("inf")

    for start in tl.range(0, K, BLOCK_SIZE):
        idx = start + offs
        mask = idx < K
        ptrs = x_ptr + row_start + idx
        x = tl.load(ptrs, mask=mask, other=-float("inf"))
        x = x.to(ACC_TYPE)
        block_max = tl.max(x, axis=0)
        cur_max = tl.maximum(cur_max, block_max)

    row_max = cur_max

    # --------------------------------------------------
    # Pass 2 – compute log(sum(exp(x - row_max)))
    # --------------------------------------------------
    sum_exp = 0.0
    for start in tl.range(0, K, BLOCK_SIZE):
        idx = start + offs
        mask = idx < K
        ptrs = x_ptr + row_start + idx
        x = tl.load(ptrs, mask=mask, other=-float("inf")).to(ACC_TYPE)
        diff = x - row_max
        exp_diff = tl.exp(diff)
        sum_exp += tl.sum(exp_diff, axis=0)

    log_sum_exp = tl.log(sum_exp)

    # --------------------------------------------------
    # Pass 3 – write normalised output
    # --------------------------------------------------
    for start in tl.range(0, K, BLOCK_SIZE):
        idx = start + offs
        mask = idx < K
        in_ptrs  = x_ptr + row_start + idx
        out_ptrs = o_ptr + row_start + idx

        x = tl.load(in_ptrs, mask=mask, other=0.0).to(ACC_TYPE)
        out = (x - row_max - log_sum_exp).to(x.dtype)
        tl.store(out_ptrs, out, mask=mask)


# ---------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------

def _log_softmax_kernel_impl(x: torch.Tensor,
                    dim: int,
                    half_to_float: bool = False) -> torch.Tensor:
    """
    Drop-in replacement for torch.ops.aten._log_softmax.default that runs
    entirely on the GPU via Triton.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (must reside on CUDA device).
    dim : int
        Dimension along which to compute log-softmax.  Negative values are
        supported (Python convention).
    half_to_float : bool, optional
        If True and `x.dtype` is fp16/bf16 the internal computation is
        up-cast to fp32 (mirrors PyTorch’s behaviour).  The final result is
        always stored in `x.dtype` regardless of this flag.
    """
    if not x.is_cuda:
        raise ValueError("Triton kernel only supports CUDA tensors")

    # ------------------ (1) Canonicalise dimension -------------------
    dim = _canonicalize_dim(dim, x.ndim)

    # ------------------ (2) Move reduction axis to last, if needed ----
    x_contig, inv_perm = _move_reduction_axis_to_last(x, dim)
    K = x_contig.shape[-1]                      # length of reduction dim
    M = x_contig.numel() // K                   # number of rows

    # ------------------ (3) Allocate output --------------------------
    out_contig = torch.empty_like(x_contig)

    # ------------------ (4) Determine accumulation dtype -------------
    if x.dtype in (torch.float16, torch.bfloat16) and half_to_float:
        acc_tl_dtype = tl.float32
    else:
        acc_tl_dtype = _torch_to_triton_dtype(x.dtype)

    # ------------------ (5) Launch Triton kernel ---------------------
    BLOCK_SIZE = 128                           # good default for most cards
    grid = (M,)

    _log_softmax_lastdim_kernel[grid](
        x_contig,                              # ptr to input
        out_contig,                            # ptr to output
        K,                                     # size of last dimension
        BLOCK_SIZE=BLOCK_SIZE,
        ACC_TYPE=acc_tl_dtype
    )

    # ------------------ (6) Restore original dimension order ---------
    if inv_perm is not None:
        out = out_contig.permute(*inv_perm).contiguous()
    else:
        out = out_contig

    return out