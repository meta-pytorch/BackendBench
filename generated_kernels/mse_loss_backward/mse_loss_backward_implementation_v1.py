import torch
import triton
import triton.language as tl


@triton.jit
def _mse_loss_backward_kernel(
    grad_out_ptr,  # pointer to grad_output (scalar or tensor)
    x_ptr,         # pointer to input
    y_ptr,         # pointer to target
    out_ptr,       # pointer to output (grad_input)
    n_elements,    # total number of elements
    scale_ptr,     # pointer to 1-element tensor containing scale (2 or 2/N)
    GRAD_IS_SCALAR: tl.constexpr,  # whether grad_out is a scalar (sum/mean)
    BLOCK_SIZE: tl.constexpr,      # block size
):
    # Program id and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    diff = x - y

    # Load grad_output: scalar or elementwise
    if GRAD_IS_SCALAR:
        go = tl.load(grad_out_ptr)  # scalar value; broadcasting in ops
    else:
        go = tl.load(grad_out_ptr + offsets, mask=mask, other=0)

    # Load scale (already incorporates reduction factor):
    # scale = 2.0 for 'none' and 'sum'
    # scale = 2.0 / N for 'mean'
    scale = tl.load(scale_ptr)

    # Compute grad_input
    # grad = grad_output * 2 * (x - y)  [* 1/N if mean]
    grad = diff * go * scale

    # Store result
    tl.store(out_ptr + offsets, grad, mask=mask)


def mse_loss_backward_kernel_impl(grad_output: torch.Tensor, input: torch.Tensor, target: torch.Tensor, reduction: int):
    """
    Triton implementation of mse_loss_backward.

    Args:
        grad_output: Tensor
            - If reduction == 0 ('none'), same shape as input/target.
            - If reduction in {1 ('mean'), 2 ('sum')}, a scalar tensor (shape []).
        input: Tensor, arbitrary shape
        target: Tensor, same shape as input
        reduction: int
            - 0 -> 'none'
            - 1 -> 'mean'
            - 2 -> 'sum'

    Returns:
        grad_input: Tensor with same shape and dtype as input.
    """
    assert input.is_cuda and target.is_cuda and grad_output.is_cuda, "All tensors must be CUDA tensors."
    assert input.shape == target.shape, "input and target must have the same shape"
    assert input.dtype == target.dtype == grad_output.dtype or grad_output.dim() == 0, \
        "Dtypes must match, except scalar grad_output is allowed."

    # Determine total number of elements and create output
    n_elements = input.numel()
    out = torch.empty_like(input)

    # Determine whether grad_out is scalar
    grad_is_scalar = (grad_output.dim() == 0) or (grad_output.numel() == 1)

    # Compute scale factor based on reduction
    # scale = 2 for 'none'/'sum', scale = 2/N for 'mean'
    if reduction == 1:  # mean
        scale_value = 2.0 / float(n_elements)
    else:  # none or sum
        scale_value = 2.0

    # Keep computations in the same dtype as input/target as much as possible
    # We'll pass scale as a 1-element tensor to control dtype precisely within the kernel
    scale_tensor = torch.tensor(scale_value, dtype=input.dtype, device=input.device)

    # Grid configuration
    BLOCK_SIZE = 1024  # power of 2 for good occupancy/coalescing
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    _mse_loss_backward_kernel[grid](
        grad_output,                 # grad_out_ptr
        input,                       # x_ptr
        target,                      # y_ptr
        out,                         # out_ptr
        n_elements,                  # n_elements
        scale_tensor,                # scale_ptr
        GRAD_IS_SCALAR=grad_is_scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out