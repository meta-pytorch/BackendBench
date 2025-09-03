import torch
import triton
import triton.language as tl


@triton.jit
def _bitwise_not_kernel(x_ptr, y_ptr, n_elements, IS_BOOL: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise bitwise NOT.

    Args:
        x_ptr:      Pointer to input tensor (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        y_ptr:      Pointer to output tensor with same dtype/shape as input
        n_elements: Total number of elements (flattened)
        IS_BOOL:    Compile-time constant indicating if dtype is boolean
        BLOCK_SIZE: Compile-time constant block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # For boolean dtype, bitwise-not must behave like logical-not.
    # Implement as x == 0 to ensure result is exactly 0 or 1, avoiding writing non-{0,1} values.
    if IS_BOOL:
        y = x == 0
    else:
        y = ~x

    tl.store(y_ptr + offsets, y, mask=mask)


def bitwise_not_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-based implementation of aten.bitwise_not.default (torch.bitwise_not).

    - Supports dtypes: bool, uint8, int8, int16, int32, int64
    - Preserves shape and dtype
    - Handles arbitrary shapes, including non-contiguous and empty tensors
    - Core computation is implemented in a Triton kernel

    Args:
        x: Input tensor on CUDA device.

    Returns:
        Tensor with same shape and dtype as x, where each element is bitwise-not of x.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if x.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"Unsupported dtype {x.dtype}. Supported: bool, uint8, int8, int16, int32, int64.")

    # Handle empty tensors early
    if x.numel() == 0:
        return torch.empty_like(x)

    # Work on contiguous buffers for coalesced memory access
    x_contig = x.contiguous()
    y_contig = torch.empty_like(x_contig)

    n_elements = x_contig.numel()

    # Configure launch: simple 1D grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _bitwise_not_kernel[grid](
        x_contig, y_contig, n_elements,
        IS_BOOL=(x.dtype == torch.bool),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape back to original shape
    return y_contig.view(x.shape)