# kernel.py
#
# Triton implementation of PyTorch’s aten.addmm.default / torch.addmm
# ===================================================================
#   out = beta * input + alpha * (mat1 @ mat2)
#
# The kernel supports:
#   • 1-D bias (length N) – broadcast to every row
#   • 2-D bias (M×N)     – element-wise add
#   • Arbitrary (even non-contiguous) strides for every tensor
#   • float16 and bfloat16 dtypes
#
# The reference test-suite (see problem statement) imports the
# `kernel_function` wrapper defined at the end of this file.

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------
#  Low-level Triton kernel
# ---------------------------------------------------------------------
@triton.jit
def _addmm_kernel(
    bias_ptr,                       # input / bias
    mat1_ptr,                       # (M, K)
    mat2_ptr,                       # (K, N)
    out_ptr,                        # (M, N) result
    M, N, K,                        # sizes
    stride_bias_row, stride_bias_col,
    stride_mat1_row, stride_mat1_col,
    stride_mat2_row, stride_mat2_col,
    stride_out_row, stride_out_col,
    alpha,                          # scalar
    beta,                           # scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,   # 1 => bias.shape = (N,)
):
    """
    Tile sizes (BLOCK_M, BLOCK_N, BLOCK_K) are compile-time constants
    supplied by the caller.  The grid is 2-D: (ceil(M/BM), ceil(N/BN)).
    """
    # --------------------------------------------------
    #   Program-ID & tile start indices
    # --------------------------------------------------
    pid_m = tl.program_id(axis=0)          # row block
    pid_n = tl.program_id(axis=1)          # col block

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Make loop variables “nice” for the compiler
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # --------------------------------------------------
    #   Blocked matrix multiplication
    # --------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in range(0, num_k_tiles):
        offs_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Pointers for current sub-tiles
        a_ptrs = mat1_ptr + (
            offs_m[:, None] * stride_mat1_row
            + offs_k[None, :] * stride_mat1_col
        )
        b_ptrs = mat2_ptr + (
            offs_k[:, None] * stride_mat2_row
            + offs_n[None, :] * stride_mat2_col
        )

        # Load with masking – out-of-bounds elements are 0
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b)                # FP32 accumulation

    # --------------------------------------------------
    #   Load & broadcast bias (beta * bias)
    # --------------------------------------------------
    if BIAS_IS_VECTOR:
        # bias: (N,)  ⇒  broadcast along rows
        bias_ptrs = bias_ptr + offs_n * stride_bias_col
        bias_vec = tl.load(bias_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        bias_tile = bias_vec[None, :]      # shape (1, BN)  – will broadcast
    else:
        # bias: (M, N)
        bias_ptrs = bias_ptr + (
            offs_m[:, None] * stride_bias_row
            + offs_n[None, :] * stride_bias_col
        )
        bias_tile = tl.load(
            bias_ptrs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

    # --------------------------------------------------
    #   Final blend   out = α * acc + β * bias
    # --------------------------------------------------
    res = alpha * acc + beta * bias_tile

    # Cast back to output dtype
    if out_ptr.dtype.element_ty == tl.float16:
        res = res.to(tl.float16)
    elif out_ptr.dtype.element_ty == tl.bfloat16:
        res = res.to(tl.bfloat16)
    else:                                 # Fallback / safety
        res = res.to(out_ptr.dtype.element_ty)

    # --------------------------------------------------
    #   Write results
    # --------------------------------------------------
    out_ptrs = out_ptr + (
        offs_m[:, None] * stride_out_row
        + offs_n[None, :] * stride_out_col
    )
    tl.store(out_ptrs, res, mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------
#  Public wrapper -----------------------------------------------------
# ---------------------------------------------------------------------
def addmm_kernel_impl(bias: torch.Tensor,
                    mat1: torch.Tensor,
                    mat2: torch.Tensor,
                    *,
                    beta: float = 1.0,
                    alpha: float = 1.0) -> torch.Tensor:
    """
    Drop-in replacement for torch.addmm implemented with Triton.

    Parameters
    ----------
    bias : Tensor[*, N]      (1-D or 2-D as in PyTorch)
    mat1 : Tensor[M, K]
    mat2 : Tensor[K, N]
    beta, alpha : scalars

    Returns
    -------
    out : Tensor[M, N]   – same dtype / device as inputs
    """
    # ----------------------------------
    #   Basic validation
    # ----------------------------------
    assert mat1.dim() == 2 and mat2.dim() == 2, "mat1 / mat2 must be 2-D"
    M, K = mat1.shape
    K2, N = mat2.shape
    assert K == K2, "mat1.shape[1] must equal mat2.shape[0]"

    assert bias.dim() in (1, 2), "bias must be 1-D or 2-D"
    if bias.dim() == 1:
        assert bias.shape[0] == N, "bias length must equal N"
    else:
        assert bias.shape == (M, N), "bias matrix must be (M, N)"

    # dtypes / device
    dtype = mat1.dtype
    assert dtype in (torch.float16, torch.bfloat16), "Only fp16 / bf16 supported"
    device = mat1.device
    mat2 = mat2.to(device)
    bias = bias.to(device)

    # ----------------------------------
    #   Allocate output
    # ----------------------------------
    out = torch.empty((M, N), dtype=dtype, device=device)

    # ----------------------------------
    #   Prepare strides & flags
    # ----------------------------------
    stride_mat1_row, stride_mat1_col = mat1.stride()
    stride_mat2_row, stride_mat2_col = mat2.stride()
    stride_out_row,  stride_out_col  = out.stride()

    if bias.dim() == 1:                   # (N,)
        stride_bias_row = 0
        stride_bias_col = bias.stride(0)
        bias_is_vector = 1
    else:                                 # (M, N)
        stride_bias_row, stride_bias_col = bias.stride()
        bias_is_vector = 0

    # ----------------------------------
    #   Launch configuration
    # ----------------------------------
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
    grid = (
        triton.cdiv(M, BLOCK_M),          # blocks along M
        triton.cdiv(N, BLOCK_N),          # blocks along N
    )

    _addmm_kernel[grid](
        bias, mat1, mat2, out,            # pointers / tensors
        M, N, K,                          # sizes
        stride_bias_row, stride_bias_col,
        stride_mat1_row, stride_mat1_col,
        stride_mat2_row, stride_mat2_col,
        stride_out_row,  stride_out_col,
        float(alpha), float(beta),        # scalars
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BIAS_IS_VECTOR=bias_is_vector,
        num_warps=8,
        num_stages=3,
    )

    return out