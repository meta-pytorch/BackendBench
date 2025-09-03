import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def _triu_kernel(
    in_ptr, out_ptr,
    B, M, N,
    stride_ib, stride_im, stride_in,
    stride_ob, stride_om, stride_on,
    diagonal,  # int
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Upper-triangular extraction kernel.

    Parameters:
      in_ptr/out_ptr: pointers to input/output
      B: number of batch matrices (product of all dims except last two)
      M, N: matrix dimensions (rows, cols)
      stride_*: element strides for input and output for [batch, row, col]
      diagonal: diagonal offset (int). 0 is main diag; >0 moves up; <0 moves down
      BLOCK_M/BLOCK_N: tile sizes for rows/cols
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Offsets within the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # In-bounds mask for MxN domain
    mask_m = offs_m < M
    mask_n = offs_n < N
    in_bounds = mask_m[:, None] & mask_n[None, :]

    # Base pointers per batch
    base_in = in_ptr + pid_b * stride_ib
    base_out = out_ptr + pid_b * stride_ob

    # Compute 2D pointers for the tile
    in_ptrs = base_in + (offs_m[:, None] * stride_im + offs_n[None, :] * stride_in)
    out_ptrs = base_out + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)

    # Load input values (mask out-of-bounds to 0)
    x = tl.load(in_ptrs, mask=in_bounds, other=0)

    # Triangular condition: keep if col >= row + diagonal
    # Broadcast to (BLOCK_M, BLOCK_N)
    keep_mask = offs_n[None, :] >= (offs_m[:, None] + diagonal)

    # Zero value of appropriate dtype
    zero = tl.zeros((BLOCK_M, BLOCK_N), dtype=out_ptr.dtype.element_ty)

    # Select values based on triangular mask; store in-bounds
    y = tl.where(keep_mask, x, zero)
    tl.store(out_ptrs, y, mask=in_bounds)


def triu_kernel_impl(x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    """
    Apply upper-triangular mask to the last two dims of `x`, equivalent to torch.triu(x, diagonal).

    Args:
      x: Input tensor with shape [..., M, N]
      diagonal: Diagonal offset (default 0). Same semantics as torch.triu:
        - 0  -> main diagonal
        - >0 -> above main diagonal
        - <0 -> below main diagonal

    Returns:
      Tensor of same shape/dtype/device as x, with entries below the specified diagonal set to 0.
    """
    if diagonal is None:
        diagonal = 0
    if x.dim() < 2:
        raise ValueError("triu requires a tensor with at least 2 dimensions")

    # Extract matrix dims
    M = x.shape[-2]
    N = x.shape[-1]

    # Early return for zero-sized dims without launching a kernel
    if M == 0 or N == 0:
        # Preserve strides/memory format to match PyTorch behavior as closely as reasonable
        out = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
        return out

    # Batch size: product of all dims except last two; if no batch dims, B=1
    if x.dim() > 2:
        B = 1
        for d in x.shape[:-2]:
            B *= int(d)
    else:
        B = 1

    # Strides for input/output. For multi-batch, we linearize batches using stride of the
    # fastest-moving batch dim (-3). This works for standard contiguous layouts and typical use.
    stride_im = x.stride(-2)
    stride_in = x.stride(-1)
    stride_ib = x.stride(-3) if x.dim() > 2 else 0

    # Allocate output; try to preserve strides of input
    out = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)

    stride_om = out.stride(-2)
    stride_on = out.stride(-1)
    stride_ob = out.stride(-3) if out.dim() > 2 else 0

    # Grid: tiles over rows, cols, and batch
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
            B,
        )

    _triu_kernel[grid](
        x, out,
        B, M, N,
        stride_ib, stride_im, stride_in,
        stride_ob, stride_om, stride_on,
        int(diagonal),
    )
    return out