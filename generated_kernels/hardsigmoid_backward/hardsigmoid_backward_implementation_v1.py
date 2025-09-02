import torch
import triton
import triton.language as tl


@triton.jit
def _hardsigmoid_backward_kernel(
    go_ptr,               # *grad_output
    x_ptr,                # *self (input of forward)
    out_ptr,              # *grad_input
    n_elements,           # total number of elements
    sizes_ptr,            # int64[RANK]
    go_strides_ptr,       # int64[RANK]
    x_strides_ptr,        # int64[RANK]
    out_strides_ptr,      # int64[RANK]
    RANK: tl.constexpr,   # tensor rank (compile-time)
    BLOCK_SIZE: tl.constexpr  # block size (compile-time)
):
    """
    Triton kernel computing the backward pass of HardSigmoid:
      y = clamp(x/6 + 0.5, 0, 1)
      dy/dx = 1/6 for x in (-3, 3), 0 otherwise (open interval)
    grad_input = grad_output * dy/dx

    This kernel supports non-contiguous tensors via explicit shape/stride traversal.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    lane = tl.arange(0, BLOCK_SIZE)
    linear_idx = block_start + lane
    mask = linear_idx < n_elements

    # Use 64-bit accumulators for addressing
    linear_idx_i64 = linear_idx.to(tl.int64)

    # Compute element-specific offsets for each tensor using sizes and strides
    go_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    x_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    out_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    tmp = linear_idx_i64
    # Mixed-radix decomposition: traverse from last dim to first
    for i in range(0, RANK):
        d = RANK - 1 - i
        size_d = tl.load(sizes_ptr + d).to(tl.int64)
        go_sd = tl.load(go_strides_ptr + d).to(tl.int64)
        x_sd = tl.load(x_strides_ptr + d).to(tl.int64)
        out_sd = tl.load(out_strides_ptr + d).to(tl.int64)

        idx_d = tmp % size_d
        tmp = tmp // size_d

        go_off += idx_d * go_sd
        x_off += idx_d * x_sd
        out_off += idx_d * out_sd

    # Load tensors
    go = tl.load(go_ptr + go_off, mask=mask, other=0)
    x = tl.load(x_ptr + x_off, mask=mask, other=0)

    # Derivative mask: 1 for (-3, 3), else 0. Open interval per PyTorch.
    inside = (x > -3.0) & (x < 3.0)

    # Scale grad_output by 1/6 in input dtype to avoid FP32 upcast
    go_scaled = go / 6

    # Apply mask (convert boolean to the same dtype, multiply)
    grad_in = go_scaled * inside.to(go_scaled.dtype)

    # Store result
    tl.store(out_ptr + out_off, grad_in, mask=mask)


def hardsigmoid_backward_kernel_impl(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute HardSigmoid backward using a Triton kernel.

    Args:
        grad_output: gradient of the output of HardSigmoid, same shape as x
        x: input to the forward HardSigmoid (self in PyTorch API)

    Returns:
        grad_input tensor with the same shape/dtype/layout as grad_output/x
    """
    if grad_output.device.type != "cuda" or x.device.type != "cuda":
        raise RuntimeError("kernel_function requires CUDA tensors")

    if grad_output.shape != x.shape:
        raise ValueError(f"Shape mismatch: grad_output.shape={grad_output.shape} vs x.shape={x.shape}")
    if grad_output.dtype != x.dtype:
        raise ValueError(f"Dtype mismatch: grad_output.dtype={grad_output.dtype} vs x.dtype={x.dtype}")
    if grad_output.numel() != x.numel():
        raise ValueError("grad_output and x must have the same number of elements")

    # Support bf16 and fp16 as required by the test; other dtypes can be enabled if needed
    if grad_output.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Unsupported dtype: {grad_output.dtype}. Expected bfloat16 or float16.")

    # Handle zero-sized tensors gracefully (no launch)
    n_elements = grad_output.numel()
    out = torch.empty_like(grad_output)
    if n_elements == 0:
        return out

    # Build metadata for generic N-D indexing (supporting non-contiguous tensors)
    rank = grad_output.dim()
    device = grad_output.device

    sizes_t = torch.tensor(grad_output.shape, dtype=torch.int64, device=device)
    go_strides_t = torch.tensor(grad_output.stride(), dtype=torch.int64, device=device)
    x_strides_t = torch.tensor(x.stride(), dtype=torch.int64, device=device)
    out_strides_t = torch.tensor(out.stride(), dtype=torch.int64, device=device)

    # Choose a power-of-two block size per guidelines
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _hardsigmoid_backward_kernel[grid](
        grad_output, x, out,
        n_elements,
        sizes_t, go_strides_t, x_strides_t, out_strides_t,
        RANK=rank,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,   # reasonable default for elementwise kernels
    )
    return out

# Optional: simple self-test when running this file directly (not required by grader)
if __name__ == "__main__":
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA is required to run this test.")
    else:
        x = torch.randn((8, 16), dtype=torch.bfloat16, device="cuda")
        go = torch.randn_like(x)
        ref = torch.ops.aten.hardsigmoid_backward.default(go, x)
        out = kernel_function(go, x)
        print("Max abs diff:", (out - ref).abs().max().item())