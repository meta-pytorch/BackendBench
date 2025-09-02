import torch
import triton
import triton.language as tl


@triton.jit
def _silu_backward_kernel(
    grad_ptr, inp_ptr, out_ptr,
    n_elements,
    s0, s1, s2, s3,             # logical shape (padded to 4D)
    g_stride0, g_stride1, g_stride2, g_stride3,   # grad_output strides
    x_stride0, x_stride1, x_stride2, x_stride3,   # input strides
    o_stride0, o_stride1, o_stride2, o_stride3,   # output strides
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for SiLU (Swish) backward:
      grad_input = grad_output * sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    The kernel supports arbitrary 1-4D shapes and arbitrary (possibly non-contiguous) strides.
    Computation is performed in FP32 for improved numerical stability and cast back to input dtype.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode flattened linear index "offsets" into 4D coordinates (d0, d1, d2, d3)
    # with the last dimension varying fastest.
    idx = offsets
    d3 = idx % s3
    idx = idx // s3
    d2 = idx % s2
    idx = idx // s2
    d1 = idx % s1
    idx = idx // s1
    d0 = idx  # remaining

    # Compute memory offsets for each tensor using provided strides
    off_g = d0 * g_stride0 + d1 * g_stride1 + d2 * g_stride2 + d3 * g_stride3
    off_x = d0 * x_stride0 + d1 * x_stride1 + d2 * x_stride2 + d3 * x_stride3
    off_o = d0 * o_stride0 + d1 * o_stride1 + d2 * o_stride2 + d3 * o_stride3

    # Load inputs
    g = tl.load(grad_ptr + off_g, mask=mask, other=0)
    x = tl.load(inp_ptr + off_x, mask=mask, other=0)

    # Upcast to FP32 for numerics
    g32 = g.to(tl.float32)
    x32 = x.to(tl.float32)

    # s = sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x32))
    # grad_input = g * s * (1 + x * (1 - s))
    grad_in = g32 * s * (1.0 + x32 * (1.0 - s))

    # Cast back to original dtype of inputs/grad (both are the same dtype in tests)
    out_val = grad_in.to(g.dtype)

    # Store result
    tl.store(out_ptr + off_o, out_val, mask=mask)


def _pack_shape_stride(t: torch.Tensor, max_dims: int = 4):
    """
    Pad shape and strides to max_dims (front-padded) so that the last axis is fastest.
    For missing leading dims, use size=1 and stride=0 (won't contribute to address).
    """
    shape = list(t.shape)
    strides = list(t.stride())
    assert len(shape) <= max_dims, "This kernel supports up to 4D tensors."
    pad = max_dims - len(shape)
    shape = [1] * pad + shape
    strides = [0] * pad + strides
    return shape, strides


def silu_backward_kernel_impl(grad_output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """
    Compute SiLU (Swish) backward using a Triton kernel.

    Args:
        grad_output: Tensor with gradient of the output, dtype in {bf16, f16}, any memory layout.
        inp:         Input tensor x that was fed to SiLU, same shape/dtype/device as grad_output.

    Returns:
        grad_input tensor with the same shape/dtype/device as grad_output.
    """
    # Basic validations
    if grad_output.device.type != "cuda" or inp.device.type != "cuda":
        raise RuntimeError("This Triton kernel requires CUDA tensors.")
    if grad_output.shape != inp.shape:
        raise ValueError("grad_output and input must have the same shape.")
    if grad_output.dtype != inp.dtype:
        raise ValueError("grad_output and input must have the same dtype.")
    if grad_output.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("Only BF16 and F16 are supported for this test.")

    # Allocate output with the same layout as grad_output
    out = torch.empty_like(grad_output)
    n_elements = grad_output.numel()

    # Handle empty tensors early (avoid launching a 0-sized grid)
    if n_elements == 0:
        return out

    # Pack shape and strides for 4D handling
    shape, g_strides = _pack_shape_stride(grad_output, 4)
    _, x_strides = _pack_shape_stride(inp, 4)
    _, o_strides = _pack_shape_stride(out, 4)

    # Triton launch configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _silu_backward_kernel[grid](
        grad_output, inp, out,
        n_elements,
        shape[0], shape[1], shape[2], shape[3],
        g_strides[0], g_strides[1], g_strides[2], g_strides[3],
        x_strides[0], x_strides[1], x_strides[2], x_strides[3],
        o_strides[0], o_strides[1], o_strides[2], o_strides[3],
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out