import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["M", "N"],
)
@triton.jit
def _tril_kernel(
    x_ptr,  # * pointer to input
    y_ptr,  # * pointer to output
    B,      # number of matrices in batch (flattened batch dims)
    M, N,   # matrix dims
    stride_b, stride_m, stride_n,  # strides for batch, row, col of x and y (identical layout)
    diagonal: tl.constexpr,        # diagonal offset (compile-time specialization not required but allowed)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program IDs: batch, row-tile, col-tile
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Cast to int64 for address arithmetic
    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_i64 = offs_n.to(tl.int64)

    # Base pointers for this batch slice
    base_x = x_ptr + pid_b.to(tl.int64) * stride_b
    base_y = y_ptr + pid_b.to(tl.int64) * stride_b

    # Compute per-element pointers
    ptrs_x = base_x + offs_m_i64[:, None] * stride_m + offs_n_i64[None, :] * stride_n
    ptrs_y = base_y + offs_m_i64[:, None] * stride_m + offs_n_i64[None, :] * stride_n

    # In-bounds mask for this tile
    in_bounds = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Triangular mask: keep lower-triangular elements (including diagonal offset)
    # Condition: j <= i + diagonal
    # offs_m, offs_n are int32; diagonal is constexpr int; broadcast safely.
    tri_mask = offs_n[None, :] <= (offs_m[:, None] + diagonal)

    # Only load elements that we will use/store
    mask_load = in_bounds & tri_mask

    # Load input values (masked)
    x_tile = tl.load(ptrs_x, mask=mask_load, other=tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1))

    # Store only the kept elements. Output tensor is pre-zeroed in the wrapper.
    tl.store(ptrs_y, x_tile, mask=mask_load)


def tril_kernel_impl(x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    """
    Compute the lower-triangular part of the last two dimensions of `x`,
    zeroing out upper-triangular elements, with optional diagonal offset.

    Args:
        x: Input tensor with shape (..., M, N). Supports 2D and batched (>=3D).
           Non-contiguous tensors are supported.
        diagonal: Which diagonal to consider:
            - 0: main diagonal
            - >0: kth diagonal above the main
            - <0: kth diagonal below the main

    Returns:
        Tensor of the same shape, dtype, and device as `x`, with upper-triangular
        elements zeroed according to the given diagonal.
    """
    assert x.ndim >= 2, "Input must be at least 2D"
    assert x.is_cuda, "Input must be on CUDA device"

    M, N = x.shape[-2], x.shape[-1]

    # Flatten all batch dims into one for kernel launch. This works for any ndim>=2.
    x_view = x.reshape(-1, M, N)

    # Output tensor initialized to zeros so we can mask stores to only the kept elements
    y = torch.zeros_like(x)
    y_view = y.reshape(-1, M, N)

    B = x_view.shape[0]

    # Strides for flattened 3D views
    stride_b = x_view.stride(0)
    stride_m = x_view.stride(1)
    stride_n = x_view.stride(2)

    # Kernel launch grid: (batch, tiles along M, tiles along N)
    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (
            B,
            triton.cdiv(M, BM),
            triton.cdiv(N, BN),
        )

    # Launch Triton kernel
    _tril_kernel[grid](
        x_view,
        y_view,
        B,
        M,
        N,
        stride_b,
        stride_m,
        stride_n,
        diagonal=diagonal,
    )
    return y