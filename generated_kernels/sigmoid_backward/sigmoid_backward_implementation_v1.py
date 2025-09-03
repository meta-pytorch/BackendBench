import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Triton kernel: sigmoid backward
# Computes: grad_input = grad_output * output * (1 - output)
# Supports non-contiguous inputs via explicit index math using sizes/strides.
# The output tensor is allocated contiguous by the Python wrapper for simplicity.
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def _sigmoid_backward_kernel(
    grad_out_ptr,  # *T
    out_ptr,       # *T
    dst_ptr,       # *T (contiguous output buffer)
    n_elements,    # int64 total number of elements
    # Shape (padded) - use up to MAX_DIMS=8
    D0, D1, D2, D3, D4, D5, D6, D7,  # int64 sizes
    # Strides for grad_out (in elements)
    Gs0, Gs1, Gs2, Gs3, Gs4, Gs5, Gs6, Gs7,  # int64
    # Strides for out (in elements)
    Os0, Os1, Os2, Os3, Os4, Os5, Os6, Os7,  # int64
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID and block offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Offsets within the flat, logical [0, n_elements) index space
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Convert to 64-bit for safe index arithmetic
    r = offs.to(tl.int64)

    # Unravel flat indices into 8D coordinates [i0..i7]
    # Note: For dims beyond the real rank, Di == 1 so idx will be 0 there.
    i7 = r % D7
    r = r // D7
    i6 = r % D6
    r = r // D6
    i5 = r % D5
    r = r // D5
    i4 = r % D4
    r = r // D4
    i3 = r % D3
    r = r // D3
    i2 = r % D2
    r = r // D2
    i1 = r % D1
    r = r // D1
    i0 = r % D0

    # Compute strided offsets for both inputs
    off_g = (
        i0 * Gs0 + i1 * Gs1 + i2 * Gs2 + i3 * Gs3 +
        i4 * Gs4 + i5 * Gs5 + i6 * Gs6 + i7 * Gs7
    )
    off_o = (
        i0 * Os0 + i1 * Os1 + i2 * Os2 + i3 * Os3 +
        i4 * Os4 + i5 * Os5 + i6 * Os6 + i7 * Os7
    )

    # Load grad_out and out with masking
    gout = tl.load(grad_out_ptr + off_g, mask=mask, other=0)
    outv = tl.load(out_ptr + off_o, mask=mask, other=0)

    # Compute in fp32 for better numerical stability, then store in dst dtype
    gout_f32 = gout.to(tl.float32)
    out_f32 = outv.to(tl.float32)

    one_minus_out = 1.0 - out_f32
    res = gout_f32 * out_f32 * one_minus_out

    # Store to contiguous output: linear offsets are just offs
    tl.store(dst_ptr + offs, res, mask=mask)


def _pad_to_max_dims(shape, strides, max_dims=8):
    """
    Pad shape and strides to max_dims with 1 and 0 respectively (0 stride not used here, but we use 1 for shapes).
    We specifically use 1 for padded sizes so that unraveling yields zeros for those dimensions.
    """
    assert len(shape) == len(strides)
    nd = len(shape)
    if nd > max_dims:
        # Flatten higher dims into leading dims, or raise. Here we raise to keep it simple and safe.
        # Tests only use up to 5D.
        raise ValueError(f"Rank {nd} > max supported dims ({max_dims}).")
    shape_padded = list(shape) + [1] * (max_dims - nd)
    strides_padded = list(strides) + [0] * (max_dims - nd)
    return shape_padded, strides_padded


def sigmoid_backward_kernel_impl(grad_out: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """
    Triton-backed implementation of sigmoid_backward:
        grad_input = grad_out * out * (1 - out)

    Args:
        grad_out: Tensor on CUDA, same shape as out
        out: Tensor on CUDA, same shape as grad_out. This is sigmoid(input) from forward.

    Returns:
        grad_input tensor (contiguous), same dtype/device/shape as grad_out.
    """
    # Basic validations
    if not (isinstance(grad_out, torch.Tensor) and isinstance(out, torch.Tensor)):
        raise TypeError("grad_out and out must be torch.Tensors")
    if grad_out.shape != out.shape:
        raise ValueError(f"Shape mismatch: grad_out.shape={grad_out.shape}, out.shape={out.shape}")
    if grad_out.device.type != "cuda" or out.device.type != "cuda":
        raise ValueError("Both grad_out and out must be CUDA tensors")
    if grad_out.dtype != out.dtype:
        raise ValueError("grad_out and out must have the same dtype")
    if grad_out.layout != torch.strided or out.layout != torch.strided:
        raise ValueError("Only strided layout is supported")

    # Allocate contiguous output buffer (PyTorch reference is also contiguous for these ops)
    result = torch.empty(grad_out.shape, dtype=grad_out.dtype, device=grad_out.device)

    n_elements = grad_out.numel()
    if n_elements == 0:
        return result  # nothing to do

    # Prepare sizes and strides (in elements). PyTorch strides are already in elements.
    shape = list(grad_out.shape)
    g_strides = list(grad_out.stride())
    o_strides = list(out.stride())

    # Pad to MAX_DIMS=8 for the kernel
    MAX_DIMS = 8
    shape_p, g_strides_p = _pad_to_max_dims(shape, g_strides, MAX_DIMS)
    _, o_strides_p = _pad_to_max_dims(shape, o_strides, MAX_DIMS)

    # Kernel launch
    # Grid is 1D over flattened elements
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _sigmoid_backward_kernel[grid](
        grad_out, out, result,
        n_elements,
        # sizes
        shape_p[0], shape_p[1], shape_p[2], shape_p[3],
        shape_p[4], shape_p[5], shape_p[6], shape_p[7],
        # grad_out strides
        g_strides_p[0], g_strides_p[1], g_strides_p[2], g_strides_p[3],
        g_strides_p[4], g_strides_p[5], g_strides_p[6], g_strides_p[7],
        # out strides
        o_strides_p[0], o_strides_p[1], o_strides_p[2], o_strides_p[3],
        o_strides_p[4], o_strides_p[5], o_strides_p[6], o_strides_p[7],
    )

    return result

"""
Notes:
- The kernel operates on flattened indices and reconstructs multi-dimensional indices via modulo/div.
- This enables correct reads from arbitrary non-contiguous source tensors using their explicit strides.
- We compute in float32 and store in the destination dtype (e.g., bfloat16) for better numerical stability.
- The wrapper returns a contiguous tensor, which matches common PyTorch behavior for such ops and
  satisfies the test's requirements (shape/dtype/device and numerical correctness).
- Boundary conditions are handled via masking on tl.load/tl.store.
- Autotuning is provided over BLOCK_SIZE, warps, and stages for different problem sizes.
"""