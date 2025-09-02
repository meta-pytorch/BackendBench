# kernel.py
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def _leaky_relu_backward_kernel(
    grad_out_ptr,     # *T, pointer to grad_output
    self_ptr,         # *T, pointer to self (either input x or output y depending on self_is_result)
    out_ptr,          # *T, pointer to output (grad_input)
    shape_ptr,        # *i32, tensor sizes [NDIMS]
    go_strides_ptr,   # *i32, grad_output strides in elements [NDIMS]
    self_strides_ptr, # *i32, self strides in elements [NDIMS]
    n_elements: tl.int32,     # total number of elements
    negative_slope,           # float scalar (passed as fp32)
    NDIMS: tl.constexpr,      # number of dimensions (compile-time constant)
    BLOCK_SIZE: tl.constexpr  # block size
):
    # Program ID
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute strided offsets for grad_output and self using row-major linearization
    # Decompose linear index into NDIMS indices and apply per-tensor strides.
    go_off = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    self_off = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    remaining = offsets

    # Iterate from last dimension to first to compute coordinates
    for d in range(NDIMS - 1, -1, -1):
        size_d = tl.load(shape_ptr + d)                   # int32
        idx_d = remaining % size_d                        # [BLOCK_SIZE], int32
        remaining = remaining // size_d                   # [BLOCK_SIZE], int32

        go_stride_d = tl.load(go_strides_ptr + d)         # int32
        self_stride_d = tl.load(self_strides_ptr + d)     # int32

        go_off += idx_d * go_stride_d
        self_off += idx_d * self_stride_d

    # Load grad_output and self
    g = tl.load(grad_out_ptr + go_off, mask=mask, other=0)
    x_or_y = tl.load(self_ptr + self_off, mask=mask, other=0)

    # Compute scaling factor: 1 if (x_or_y > 0), else negative_slope
    # Note: When self_is_result=True, x_or_y is y; y>0 <=> x>0 for positive slopes (common case).
    one = tl.full([BLOCK_SIZE], 1.0, dtype=g.dtype)
    slope = tl.full([BLOCK_SIZE], negative_slope, dtype=g.dtype)
    scale = tl.where(x_or_y > 0, one, slope)

    # Multiply by grad_output to get grad_input
    out = g * scale

    # Store result contiguously for good coalescing
    tl.store(out_ptr + offsets, out, mask=mask)


def leaky_relu_backward_kernel_impl(grad_output: torch.Tensor,
                    self_tensor: torch.Tensor,
                    negative_slope: float,
                    self_is_result: bool) -> torch.Tensor:
    """
    Triton implementation of aten.leaky_relu_backward.default.

    Args:
        grad_output: Tensor with upstream gradients (same shape as self_tensor).
        self_tensor: Tensor that is either the original input x (self_is_result=False)
                     or the forward result y = leaky_relu(x) (self_is_result=True).
        negative_slope: float negative slope used in leaky ReLU.
        self_is_result: bool flag indicating whether self_tensor is the forward result.

    Returns:
        grad_input tensor with the same shape, dtype, and device as grad_output.

    Notes:
        - The kernel computes grad_input = grad_output * (1 if ref > 0 else negative_slope),
          where ref is self_tensor (x or y depending on self_is_result).
        - Handles arbitrary shapes and strides for inputs.
        - Output is stored contiguously for optimal performance.
    """
    # Basic checks
    assert grad_output.shape == self_tensor.shape, "Shapes of grad_output and self must match"
    assert grad_output.device.type == "cuda" and self_tensor.device.type == "cuda", "Tensors must be on CUDA"
    assert grad_output.dtype == self_tensor.dtype, "Dtypes of grad_output and self must match"

    device = grad_output.device
    dtype = grad_output.dtype
    shape = tuple(grad_output.shape)
    ndims = len(shape)
    assert ndims >= 1, "Zero-dimensional tensors are not supported"

    # Allocate output (contiguous for better store coalescing)
    out = torch.empty_like(grad_output, memory_format=torch.contiguous_format)

    # Prepare metadata on device (sizes and strides as int32 in elements)
    sizes_i32 = torch.tensor(shape, device=device, dtype=torch.int32)
    go_strides_i32 = torch.tensor(grad_output.stride(), device=device, dtype=torch.int32)
    self_strides_i32 = torch.tensor(self_tensor.stride(), device=device, dtype=torch.int32)

    n_elements = out.numel()

    # Grid: 1D launch
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel. We do not need different logic for self_is_result inside the kernel;
    # we simply use self_tensor as the reference for sign.
    _leaky_relu_backward_kernel[grid](
        grad_output,
        self_tensor,
        out,
        sizes_i32,
        go_strides_i32,
        self_strides_i32,
        n_elements,
        float(negative_slope),
        NDIMS=ndims,
    )

    return out