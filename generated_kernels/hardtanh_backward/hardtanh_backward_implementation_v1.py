import torch
import triton
import triton.language as tl


# Autotune configurations for elementwise kernel
_configs = [
    triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_configs, key=["n_elements"])
@triton.jit
def _hardtanh_backward_kernel(
    grad_out_ptr,  # *T: pointer to grad_output tensor
    x_ptr,         # *T: pointer to input tensor (self)
    out_ptr,       # *T: pointer to output grad_input tensor
    n_elements,    # int: total number of elements
    min_val,       # float32 scalar: lower bound
    max_val,       # float32 scalar: upper bound
    BLOCK_SIZE: tl.constexpr,  # compile-time constant for vectorized processing
):
    """
    Triton kernel to compute hardtanh backward:
      grad_input[i] = grad_output[i] if (x[i] strictly between (min_val, max_val)) or x[i] is NaN, else 0

    Notes:
    - PyTorch's aten.hardtanh_backward uses strict inequalities and propagates gradient for NaN inputs.
    - Operates on flattened memory; boundary masking handles the tail.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    go = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Promote to fp32 for comparisons to match PyTorch semantics.
    x_f32 = x.to(tl.float32)

    # Strict inequalities for gradient pass-through.
    in_open_interval = (x_f32 > min_val) & (x_f32 < max_val)
    # Propagate gradient for NaN inputs: NaN != NaN
    is_nan = x_f32 != x_f32
    cond = in_open_interval | is_nan

    zero = tl.zeros_like(go)
    result = tl.where(cond, go, zero)

    tl.store(out_ptr + offsets, result, mask=mask)


def hardtanh_backward_kernel_impl(grad_output: torch.Tensor, inp: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Wrapper to launch the Triton hardtanh backward kernel.

    Args:
      grad_output: PyTorch tensor containing upstream gradients.
      inp:         PyTorch tensor containing the forward pass input (same shape as grad_output).
      min_val:     Lower bound for hardtanh (exclusive for backward).
      max_val:     Upper bound for hardtanh (exclusive for backward).

    Returns:
      A tensor grad_input with the same shape and dtype as grad_output, where:
        grad_input = grad_output if (inp in (min_val, max_val)) or isnan(inp) else 0.
    """
    if grad_output.device.type != "cuda" or inp.device.type != "cuda":
        raise RuntimeError("This kernel requires CUDA tensors.")

    if grad_output.shape != inp.shape:
        raise ValueError(f"Shape mismatch: grad_output.shape={grad_output.shape}, inp.shape={inp.shape}")

    if grad_output.dtype != inp.dtype:
        raise ValueError(f"Dtype mismatch: grad_output.dtype={grad_output.dtype}, inp.dtype={inp.dtype}")

    # Supported dtypes: float16, bfloat16, float32
    if grad_output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype: {grad_output.dtype}. Supported: float16, bfloat16, float32")

    # Make inputs contiguous for coalesced access and to support arbitrary layouts
    go = grad_output.contiguous()
    x = inp.contiguous()

    # Allocate output contiguous
    out = torch.empty_like(go)

    n_elements = go.numel()

    # Define launch grid; 1D launch over flattened elements
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Cast bounds to float32 for the kernel (kernel compares in fp32)
    min_val_f32 = float(min_val)
    max_val_f32 = float(max_val)

    _hardtanh_backward_kernel[grid](
        go, x, out,
        n_elements,
        min_val_f32, max_val_f32,
    )

    return out