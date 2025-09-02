import torch
import triton
import triton.language as tl


@triton.jit
def _hardswish_backward_kernel(
    go_ptr, x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel computing the gradient of HardSwish elementwise:
      grad_input = grad_output * dhardswish(x)

    dhardswish(x) =
      0                         if x <= -3
      1                         if x >= 3
      x / 3 + 0.5               otherwise

    Computation is done in float32 for stability and cast back to the
    output dtype on store.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs with masking to handle boundaries
    go = tl.load(go_ptr + offsets, mask=mask, other=0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Upcast to fp32 for compute
    go_f32 = go.to(tl.float32)
    x_f32 = x.to(tl.float32)

    # Piecewise derivative of HardSwish
    three = 3.0
    half = 0.5
    third = 1.0 / 3.0

    cond_lo = x_f32 <= -three
    cond_hi = x_f32 >= three
    grad_mid = x_f32 * third + half
    grad = tl.where(cond_hi, 1.0, tl.where(cond_lo, 0.0, grad_mid))

    # Chain rule
    res_f32 = go_f32 * grad

    # Cast back to output dtype and store
    out_dtype = out_ptr.dtype.element_ty
    res = res_f32.to(out_dtype)
    tl.store(out_ptr + offsets, res, mask=mask)


def hardswish_backward_kernel_impl(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of HardSwish using a Triton kernel.

    Args:
        grad_output: Upstream gradient tensor, same shape as x.
        x: Input tensor ("self") to HardSwish, same shape as grad_output.

    Returns:
        grad_input tensor (same shape/dtype/device as inputs).
    """
    if not (isinstance(grad_output, torch.Tensor) and isinstance(x, torch.Tensor)):
        raise TypeError("kernel_function expects torch.Tensor inputs for (grad_output, x).")
    if grad_output.shape != x.shape:
        raise ValueError(f"Shape mismatch: grad_output.shape={grad_output.shape}, x.shape={x.shape}")
    if grad_output.device.type != "cuda" or x.device.type != "cuda":
        raise RuntimeError("Inputs must be CUDA tensors.")
    if grad_output.dtype != x.dtype:
        raise ValueError(f"Dtype mismatch: grad_output.dtype={grad_output.dtype}, x.dtype={x.dtype}")

    # Make contiguous copies to ensure coalesced memory access.
    go_c = grad_output.contiguous()
    x_c = x.contiguous()

    # Allocate output tensor
    out = torch.empty_like(go_c)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Kernel launch configuration
    BLOCK_SIZE = 1024  # power-of-two for good occupancy/coalescing
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _hardswish_backward_kernel[grid](
        go_c, x_c, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return out