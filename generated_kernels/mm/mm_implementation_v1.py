import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Tiled matrix multiplication kernel: C[M, N] = A[M, K] @ B[K, N]

    - A has shape (M, K) with strides (stride_am, stride_ak)
    - B has shape (K, N) with strides (stride_bk, stride_bn)
    - C has shape (M, N) with strides (stride_cm, stride_cn)

    Accumulates in fp32 and stores back to the output dtype.
    Handles non-contiguous inputs via explicit strides and masks for boundaries.
    """
    tl.static_assert(BLOCK_SIZE_M % 16 == 0)
    tl.static_assert(BLOCK_SIZE_N % 16 == 0)
    tl.static_assert(BLOCK_SIZE_K % 16 == 0)

    # Program IDs for the 2D grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute the block ranges for M and N
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # Pointers to the first tiles of A and B for this block
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Number of K tiles
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    # Main loop over K
    for kt in range(0, k_tiles):
        k_start = kt * BLOCK_SIZE_K
        k_mask_a = (offs_m[:, None] < M) & (k_start + offs_k[None, :] < K)
        k_mask_b = (k_start + offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)

        # Matrix multiply update
        acc = tl.dot(a, b, acc)

        # Advance to next K tile
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Cast accumulator to output dtype and store
    out = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, out, mask=c_mask)


def mm_kernel_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton-based matrix multiplication equivalent to torch.mm for 2D inputs.

    Args:
        A: Tensor of shape (M, K), dtype in {torch.bfloat16, torch.float16, torch.float32}
        B: Tensor of shape (K, N), same dtype as A

    Returns:
        Tensor C of shape (M, N) with same dtype and device as A/B.

    Notes:
        - Supports non-contiguous inputs via explicit strides.
        - Accumulates in fp32 for numerical stability and casts back to input dtype.
        - Properly handles boundary conditions and zero-sized dimensions.
    """
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("kernel_function only supports 2D matrices.")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes for mm: {A.shape} @ {B.shape}")
    if A.dtype != B.dtype:
        raise ValueError(f"Dtype mismatch: A.dtype={A.dtype}, B.dtype={B.dtype}")
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("Inputs must be CUDA tensors.")
    if A.device != B.device:
        raise ValueError("Inputs must be on the same device.")

    M, K = A.shape
    Kb, N = B.shape
    dtype = A.dtype
    device = A.device

    # Early exit for zero-sized dimensions to avoid launching a 0-grid kernel
    if M == 0 or N == 0 or K == 0:
        return torch.empty((M, N), dtype=dtype, device=device)

    # Allocate output
    C = torch.empty((M, N), dtype=dtype, device=device)

    # Compute grid size
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    _mm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    return C