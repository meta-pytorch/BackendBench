import torch
import triton
import triton.language as tl


@triton.jit
def _any_reduce_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Block-wise reduction kernel that computes whether any element in a chunk
    is non-zero (or True). Produces one int32 flag per program:
      - 1 if any element in the block is non-zero
      - 0 otherwise

    Args:
        x_ptr: pointer to input tensor data (can be any numeric or bool dtype)
        out_ptr: pointer to output int32 tensor for partial results
        n_elements: total number of elements to process
        BLOCK_SIZE: compile-time constant controlling block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load a block of data; out-of-bounds entries are 0 so they don't affect "any"
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    zero = tl.zeros_like(x)

    # Compare against zero; for floats, NaN != 0 is True as desired
    pred = x != zero

    # Reduce: any() over the block -> max over 0/1
    pred_i32 = pred.to(tl.int32)
    block_any = tl.max(pred_i32, axis=0)

    # Write one int32 flag per block/program
    tl.store(out_ptr + pid, block_any)


def any_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-based implementation of aten.any.default.

    Args:
        x: Input tensor of any numeric or bool dtype, any shape, on CUDA device.

    Returns:
        0-dim boolean tensor on the same device indicating whether any element is non-zero (or True).
    """
    if not torch.is_tensor(x):
        raise TypeError("kernel_function expects a torch.Tensor as input.")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if x.numel() == 0:
        # By PyTorch semantics, any on empty returns False
        return torch.tensor(False, device=x.device, dtype=torch.bool)

    # For simplicity and performance, operate on a contiguous buffer.
    # This does not compute the result; it's only a layout conversion.
    x_in = x if x.is_contiguous() else x.contiguous()

    n_elements = x_in.numel()
    device = x_in.device

    # First pass: reduce input to block-wise partials (int32 flags)
    # Choose a reasonable block size; autotuning could be added if desired.
    BLOCK_SIZE = 4096
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial = torch.empty((num_blocks,), dtype=torch.int32, device=device)

    grid = (num_blocks,)
    _any_reduce_kernel[grid](x_in, partial, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Subsequent passes: keep reducing the int32 partials until one value remains.
    while partial.numel() > 1:
        n = partial.numel()
        num_blocks = triton.cdiv(n, BLOCK_SIZE)
        next_partial = torch.empty((num_blocks,), dtype=torch.int32, device=device)
        grid = (num_blocks,)
        _any_reduce_kernel[grid](partial, next_partial, n, BLOCK_SIZE=BLOCK_SIZE)
        partial = next_partial

    # Convert the final int32 flag to a 0-dim bool tensor on the same device
    result = (partial[0] != 0)
    return result  # 0-dim bool tensor on device