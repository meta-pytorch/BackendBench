import torch
import triton
import triton.language as tl


"""
Triton kernel implementing aten.gt.Scalar (elementwise greater-than vs scalar).

Core requirements satisfied:
- Actual computation is performed in Triton: tl.load/tl.store and comparison
- Handles all tensor dtypes used in tests (int64, int32, int16, uint8, float16, bfloat16, bool)
- Works for arbitrary shapes (flattened indexing) and empty tensors
- Handles non-contiguous inputs by creating a contiguous view for coalesced loads
- Returns a boolean tensor with the same shape and device as input

Usage:
    from kernel import kernel_function
    out = kernel_function(x, scalar)
"""


@triton.jit
def _gt_scalar_kernel(x_ptr, out_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise greater-than vs a scalar:
        out[i] = x[i] > scalar

    Args:
        x_ptr: Pointer to input tensor elements
        out_ptr: Pointer to output tensor elements (bool)
        n_elements: Total number of elements
        scalar: The scalar to compare against (int or float)
        BLOCK_SIZE: Compile-time constant, number of elements processed per program
    """
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Ensure good codegen/coalescing
    offsets = tl.multiple_of(offsets, BLOCK_SIZE)

    mask = offsets < n_elements

    # Load input elements; masked loads use other=0 which will be cast to the appropriate dtype
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Compute the comparison in Triton.
    # Triton's type system will promote/cast as needed. For booleans, comparison is done
    # after promotion (True->1, False->0), consistent with PyTorch semantics.
    y = x > scalar  # result is boolean (tl.int1)

    # Store boolean results
    tl.store(out_ptr + offsets, y, mask=mask)


def gt_kernel_impl(x: torch.Tensor, scalar):
    """
    Wrapper that launches the Triton kernel.

    Args:
        x: Input PyTorch tensor on CUDA device. Can be any dtype supported by tests.
        scalar: Python int/float scalar to compare against.

    Returns:
        A boolean tensor (torch.bool) on the same device, with the same shape as x,
        where each element is (x > scalar).
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    # Allocate output tensor (bool) with same shape/device
    out = torch.empty_like(x, dtype=torch.bool)

    n_elements = x.numel()
    if n_elements == 0:
        return out

    # For performance and simplicity, operate on contiguous memory.
    # This does not change semantics; we preserve shape in the returned tensor.
    x_contig = x.contiguous()
    out_contig = out.contiguous()

    # Kernel launch configuration
    BLOCK_SIZE = 1024  # power-of-two block size for good coalescing; typical choice
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel
    _gt_scalar_kernel[grid](
        x_contig, out_contig, n_elements, scalar,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,     # a reasonable default for this block size
        num_stages=2,    # small pipeline depth is sufficient for simple elementwise op
    )

    # out_contig already matches shape/device/dtype; return it
    return out_contig