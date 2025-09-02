import torch
import triton
import triton.language as tl

"""
Triton implementation of aten.threshold_backward.default

Semantics:
  grad_input = grad_output where NOT(self <= threshold), else 0

Important NaN semantics:
- In PyTorch's aten.threshold_backward.default, the mask used is (self <= threshold).
  Since (NaN <= threshold) is False, NaNs in `self` do NOT get zeroed and thus
  their gradients are propagated (kept). This differs from using (self > threshold),
  which would zero out NaNs. We therefore implement the mask as ~(self <= threshold).

Notes:
- The kernel operates elementwise over a flattened, contiguous view for coalesced access.
- The wrapper accepts arbitrary input layouts (contiguous, non-contiguous, channels_last).
- Computation happens in the input dtype; no upcasting is performed.
"""

# Autotune configurations for different problem sizes
_threshold_configs = [
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_threshold_configs, key=["n_elements"])
@triton.jit
def _threshold_backward_kernel(
    grad_out_ptr,     # *T
    inp_ptr,          # *T
    out_ptr,          # *T
    n_elements,       # int32
    threshold_f32,    # float32 scalar
    BLOCK_SIZE: tl.constexpr,
):
    """
    Elementwise kernel:
      out[i] = grad_out[i] if NOT (inp[i] <= threshold) else 0
    This matches PyTorch's aten.threshold_backward.default NaN semantics.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    grad = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    x = tl.load(inp_ptr + offsets, mask=mask, other=0)

    # Build a vector threshold in the same dtype as x to avoid precision surprises
    thr = tl.full([BLOCK_SIZE], threshold_f32, dtype=x.dtype)

    # Keep gradient where NOT (x <= thr)
    # This ensures NaNs in x keep gradient: (NaN <= thr) -> False, negation -> True
    keep = ~(x <= thr)

    zeros = tl.zeros([BLOCK_SIZE], dtype=grad.dtype)
    out = tl.where(keep, grad, zeros)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def threshold_backward_kernel_impl(grad_output: torch.Tensor, inp: torch.Tensor, threshold: float):
    """
    Compute the backward of threshold in Triton:
      grad_input = grad_output where NOT(self <= threshold), else 0

    Args:
      grad_output: torch.Tensor on CUDA
      inp: torch.Tensor on CUDA (same shape and dtype as grad_output)
      threshold: Python float

    Returns:
      torch.Tensor with same shape and dtype as inputs, on CUDA.
    """
    if not (isinstance(grad_output, torch.Tensor) and isinstance(inp, torch.Tensor)):
        raise TypeError("grad_output and inp must be torch.Tensor")
    if not grad_output.is_cuda or not inp.is_cuda:
        raise ValueError("grad_output and inp must be CUDA tensors")
    if grad_output.shape != inp.shape:
        raise ValueError(f"Shape mismatch: grad_output.shape={grad_output.shape}, inp.shape={inp.shape}")
    if grad_output.dtype != inp.dtype:
        raise ValueError(f"Dtype mismatch: grad_output.dtype={grad_output.dtype}, inp.dtype={inp.dtype}")

    # Flattened contiguous views for computation
    go_contig = grad_output.contiguous()
    x_contig = inp.contiguous()

    # Output contiguous buffer
    out_contig = torch.empty_like(go_contig)

    n_elements = go_contig.numel()
    if n_elements == 0:
        return torch.empty_like(grad_output)

    # Define launch grid
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    _threshold_backward_kernel[grid](
        go_contig, x_contig, out_contig,
        n_elements,
        float(threshold),
    )

    # Create result with the same logical shape and layout as grad_output
    result = torch.empty_like(grad_output)
    result.copy_(out_contig)
    return result