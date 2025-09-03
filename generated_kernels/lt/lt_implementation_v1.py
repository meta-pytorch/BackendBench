import triton
import triton.language as tl
import torch


@triton.jit
def _lt_scalar_kernel(x_ptr, s_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise less-than (Tensor < Scalar) kernel.

    Args:
        x_ptr: pointer to input tensor (any supported dtype)
        s_ptr: pointer to a 0-dim tensor containing the scalar converted to x's dtype
        out_ptr: pointer to output tensor (torch.bool)
        n_elements: total number of elements to process
        BLOCK_SIZE: compile-time block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load tensor block
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Load scalar already cast to x's dtype; broadcast in comparison
    s = tl.load(s_ptr)

    # Elementwise comparison; result is boolean mask
    y = x < s

    # Store results (bool)
    tl.store(out_ptr + offsets, y, mask=mask)


def lt_kernel_impl(x: torch.Tensor, scalar):
    """
    Implements aten.lt.Scalar (tensor < scalar) using a Triton kernel.

    - Works for float (bf16/half), integer (including uint8), and bool tensors
    - Handles non-contiguous inputs by operating on a contiguous copy
    - Supports empty and 0-dim tensors
    - Returns a torch.bool tensor with the same shape as x

    Args:
        x: input torch.Tensor on CUDA
        scalar: Python scalar (int/float/bool)

    Returns:
        torch.Tensor (dtype=torch.bool) with the same shape as x
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if x.numel() == 0:
        # Handle empty tensors without launching a kernel
        return torch.empty_like(x, dtype=torch.bool)

    # Work on a contiguous view to ensure coalesced memory access.
    # This does not change values, only layout.
    x_contig = x.contiguous()

    # Prepare output (contiguous)
    out = torch.empty_like(x_contig, dtype=torch.bool)

    # Create a device scalar buffer cast to the same dtype as x.
    # This mirrors PyTorch's aten.lt.Scalar semantics where the scalar is
    # converted to the tensor's dtype before comparison.
    s_buf = torch.empty((), dtype=x_contig.dtype, device=x_contig.device)
    s_buf.fill_(scalar)

    # Launch configuration
    n_elements = x_contig.numel()
    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    _lt_scalar_kernel[grid](x_contig, s_buf, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Return with the original logical shape (strides are not required to match)
    return out.view(x.shape)