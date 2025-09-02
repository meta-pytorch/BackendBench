import triton
import triton.language as tl
import torch


"""
Triton kernel: elementwise maximum with full broadcasting and non-contiguous support.

- Implements aten.maximum.default behavior for matching dtypes (tested: bfloat16, int32).
- Supports:
  * Broadcasting (including 0-dim scalars)
  * Arbitrary ranks (up to MAX_RANK)
  * Non-contiguous inputs via explicit stride arithmetic
- The computation is performed in Triton (no PyTorch math inside the kernel).
- The wrapper `kernel_function` prepares shapes/strides and launches the kernel.
"""

# Chosen defaults
_DEFAULT_BLOCK_SIZE = 1024
_MAX_RANK = 8  # Support up to 8D tensors. Can be increased if needed.


@triton.jit
def _maximum_broadcast_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements,             # total number of output elements (int32)
    shape_ptr,              # int32[MAX_RANK]
    astride_ptr,            # int32[MAX_RANK]
    bstride_ptr,            # int32[MAX_RANK]
    BLOCK_SIZE: tl.constexpr,
    MAX_RANK: tl.constexpr,
):
    # Program ID and linear offsets for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vector [BLOCK_SIZE]
    mask = offsets < n_elements

    # Prepare linear indices into a and b using broadcasted strides
    # We decompose the linear output index into multi-dimensional coordinates
    # using the broadcasted output shape. For each dim d, idx_d = (idx // prod(shape[d+1:])) % shape[d].
    # Then: a_offset = sum(idx_d * astride[d]), b_offset = sum(idx_d * bstride[d]).
    # Arrays are right-aligned into MAX_RANK, and shape[d] == 1 implies broadcast (stride 0).

    # Use int32 arithmetic (sufficient for test sizes). If necessary, switch to int64.
    idx = offsets.to(tl.int32)
    a_lin = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    b_lin = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # Iterate from last dimension to first (row-major linearization)
    # shape_ptr[d], astride_ptr[d], bstride_ptr[d] are scalars; operations broadcast over the vector idx
    for d in range(MAX_RANK - 1, -1, -1):
        s = tl.load(shape_ptr + d)          # int32 scalar: output shape at dim d
        # When s == 1, rem will be 0 and idx won't change (idx //= 1), which is correct.
        rem = tl.where(s != 0, idx % s, 0)  # guard s==0 (shouldn't happen) to avoid div by zero
        idx = tl.where(s != 0, idx // s, idx)

        astr = tl.load(astride_ptr + d)     # int32 scalar
        bstr = tl.load(bstride_ptr + d)     # int32 scalar
        a_lin += rem * astr
        b_lin += rem * bstr

    # Load values with masking for threads beyond n_elements
    a_val = tl.load(a_ptr + a_lin, mask=mask, other=0)
    b_val = tl.load(b_ptr + b_lin, mask=mask, other=0)

    # Elementwise maximum using Triton ops (works for floats and ints)
    res = tl.where(a_val > b_val, a_val, b_val)

    # Store result
    tl.store(out_ptr + offsets, res, mask=mask)


def _compute_broadcast_shape(shape_a, shape_b):
    """
    Compute the broadcasted shape following PyTorch/Numpy rules:
    - Align from the right
    - Each dimension must match or one of them must be 1
    """
    ra, rb = len(shape_a), len(shape_b)
    r = max(ra, rb)
    out = []
    for i in range(1, r + 1):
        da = shape_a[-i] if i <= ra else 1
        db = shape_b[-i] if i <= rb else 1
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {shape_a} and {shape_b}")
    return tuple(reversed(out))


def _aligned_strides_for_broadcast(tensor, out_shape):
    """
    Given a tensor and the target broadcasted out_shape, produce the per-dimension strides
    (in elements) aligned to out_shape, applying stride=0 for broadcasted dimensions.
    """
    in_shape = list(tensor.shape)
    in_stride = list(tensor.stride())  # strides are in elements
    r_out = len(out_shape)
    r_in = len(in_shape)

    aligned = [0] * r_out
    leading = r_out - r_in  # number of leading dims to pad on the left

    for i in range(r_out):
        if i < leading:
            # Dimension does not exist in input -> broadcast
            aligned[i] = 0
        else:
            j = i - leading
            size_in = in_shape[j]
            if size_in == 1:
                # Broadcast along this dimension
                aligned[i] = 0
            else:
                # Must match out dim (already validated)
                aligned[i] = in_stride[j]
    return aligned


def maximum_kernel_impl(a, b, *, block_size=_DEFAULT_BLOCK_SIZE, max_rank=_MAX_RANK):
    """
    Wrapper function that prepares metadata and launches the Triton kernel.

    Args:
      a: PyTorch tensor on CUDA device (supports 0-D to 8-D). Dtype tested: bfloat16, int32.
      b: PyTorch tensor on CUDA device (supports 0-D to 8-D). Must have same dtype as 'a' in tests.
      block_size: Triton block size (power of two recommended).
      max_rank: Maximum rank supported by the kernel (default 8).

    Returns:
      out: Result tensor with broadcasted shape, same dtype/device as inputs.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("kernel_function expects PyTorch tensors as inputs.")

    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError("Both inputs must be CUDA tensors.")

    if a.dtype != b.dtype:
        # The tests use matching dtypes. We keep this simple.
        raise TypeError(f"Dtype mismatch: a.dtype={a.dtype}, b.dtype={b.dtype}")

    device = a.device
    if b.device != device:
        raise RuntimeError("Inputs must be on the same device.")

    # Compute broadcasted output shape
    out_shape = _compute_broadcast_shape(tuple(a.shape), tuple(b.shape))
    out = torch.empty(out_shape, dtype=a.dtype, device=device)

    # Short-circuit: nothing to do
    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Prepare aligned strides for broadcasting and pack into MAX_RANK tensors
    if len(out_shape) > max_rank:
        raise ValueError(f"Output rank {len(out_shape)} exceeds supported MAX_RANK={max_rank}")

    a_strides = _aligned_strides_for_broadcast(a, out_shape)
    b_strides = _aligned_strides_for_broadcast(b, out_shape)

    # Prepare shape/stride arrays, right-aligned into MAX_RANK
    pad = max_rank - len(out_shape)
    shape_full = ([1] * pad) + list(out_shape)
    a_strides_full = ([0] * pad) + a_strides
    b_strides_full = ([0] * pad) + b_strides

    # Convert to device tensors (int32) for kernel consumption
    shape_dev = torch.tensor(shape_full, dtype=torch.int32, device=device)
    a_stride_dev = torch.tensor(a_strides_full, dtype=torch.int32, device=device)
    b_stride_dev = torch.tensor(b_strides_full, dtype=torch.int32, device=device)

    # Compute launch grid
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    _maximum_broadcast_kernel[grid](
        a, b, out,
        n_elements,
        shape_dev, a_stride_dev, b_stride_dev,
        BLOCK_SIZE=block_size,
        MAX_RANK=max_rank,
    )

    return out


# Optional: provide a simple manual test when running this file directly.
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available.")
    else:
        torch.manual_seed(0)
        device = "cuda"

        # Quick sanity checks
        a = torch.randn((4, 12, 128, 128), dtype=torch.bfloat16, device=device)
        b = torch.tensor(0.0, dtype=torch.bfloat16, device=device)
        ref = torch.maximum(a, b)
        out = kernel_function(a, b)
        print("BF16 broadcast scalar:", torch.allclose(out, ref, rtol=1e-2, atol=1e-2))

        a = torch.randint(-100, 100, (513,), dtype=torch.int32, device=device)
        b = torch.tensor(-10, dtype=torch.int32, device=device)
        ref = torch.maximum(a, b)
        out = kernel_function(a, b)
        print("int32 broadcast scalar:", torch.equal(out, ref))

        base = torch.randn((17, 33, 16), dtype=torch.bfloat16, device=device)
        a = base.transpose(0, 2)  # (16, 33, 17), non-contiguous
        b = torch.randn((1, 33, 1), dtype=torch.bfloat16, device=device)
        ref = torch.maximum(a, b)
        out = kernel_function(a, b)
        print("BF16 non-contiguous + broadcast:", torch.allclose(out, ref, rtol=1e-2, atol=1e-2))