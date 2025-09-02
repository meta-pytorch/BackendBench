import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_kernel(
    x_ptr, out_ptr, n_elements,
    min_val, max_val,
    HAS_MIN: tl.constexpr, HAS_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Elementwise clamp kernel:
      out[i] = min(max(x[i], min_val), max_val)
    with optional min/max (if HAS_MIN/HAS_MAX are false, those bounds are ignored).

    Notes:
    - Supports integer and floating dtypes (including bfloat16 / float16).
    - NaN handling: comparisons with NaN are false, so NaNs propagate unchanged.
    - Proper masking for OOB protection.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input (masked for OOB lanes)
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Clamp
    y = x
    if HAS_MIN:
        minv = tl.full((BLOCK_SIZE,), min_val, x.dtype)
        y = tl.where(x < minv, minv, y)
    if HAS_MAX:
        maxv = tl.full((BLOCK_SIZE,), max_val, x.dtype)
        y = tl.where(y > maxv, maxv, y)

    # Store result (masked)
    tl.store(out_ptr + offsets, y, mask=mask)


def clamp_kernel_impl(x: torch.Tensor, min=None, max=None) -> torch.Tensor:
    """
    Triton implementation of torch.clamp(x, min=min, max=max).

    Args:
        x: CUDA tensor. Supported dtypes include bfloat16, float16, int8, int32, etc.
        min: Optional scalar lower bound (Python int/float). If None, lower bound is ignored.
        max: Optional scalar upper bound (Python int/float). If None, upper bound is ignored.

    Returns:
        A new tensor with the same shape and dtype as x, with values clamped to [min, max].

    Notes:
        - Matches PyTorch semantics, including raising an error if both min and max are None.
        - Does not raise when min > max; behavior matches torch.clamp:
          result equals max everywhere because upper clamp is applied after lower clamp.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    # Match PyTorch behavior: both None is an error
    if (min is None) and (max is None):
        raise ValueError("torch.clamp: at least one of 'min' or 'max' must not be None")

    # Work on a contiguous view for coalesced memory access
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig, memory_format=torch.contiguous_format)

    n_elements = x_contig.numel()
    if n_elements == 0:
        return out.reshape(x.shape)

    # Kernel launch configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Fallback scalar values if None; not used when HAS_MIN/HAS_MAX are False
    min_val = 0 if min is None else min
    max_val = 0 if max is None else max

    _clamp_kernel[grid](
        x_contig, out,
        n_elements,
        min_val, max_val,
        HAS_MIN=(min is not None),
        HAS_MAX=(max is not None),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return out.reshape(x.shape)


if __name__ == "__main__":
    if torch.cuda.is_available():
        t = (torch.randn((16, 16), device="cuda", dtype=torch.bfloat16) * 3)
        y_ref = torch.clamp(t, -0.5, 0.5)
        y = kernel_function(t, -0.5, 0.5)
        print("Allclose:", torch.allclose(y_ref, y, rtol=1e-2, atol=1e-2))
    else:
        print("CUDA not available")