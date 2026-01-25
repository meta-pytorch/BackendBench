import torch
import triton
import triton.language as tl


@triton.jit
def _fill_tensor_kernel(x_ptr, src_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    In-place fill kernel.
    - x_ptr: pointer to tensor data to be filled
    - src_ptr: pointer to a 0-d tensor holding the scalar fill value
    - n_elements: total number of elements to fill
    - BLOCK_SIZE: compile-time constant specifying block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load scalar once per program and cast to destination dtype
    val = tl.load(src_ptr)
    val = val.to(x_ptr.dtype.element_ty)

    # Broadcast scalar to a vector for masked store
    vals = tl.full((BLOCK_SIZE,), val, dtype=x_ptr.dtype.element_ty)

    # Coalesced, masked store
    tl.store(x_ptr + offsets, vals, mask=mask)


def fill___Tensor_kernel_impl(x: torch.Tensor, src: torch.Tensor):
    """
    In-place fill of tensor x with the scalar value contained in src.

    - Fused stages: This operator is a single in-place fill, so there is no meaningful multi-op
      pipeline to fuse. The entire computation is handled in a single Triton kernel.

    Runtime behavior:
    - Validates inputs (device, dtype compatibility, contiguity).
    - Launches a Triton kernel that writes the scalar into every element of x.
    - Returns the same tensor x (mutated in-place), matching aten.fill_.Tensor semantics.

    Notes:
    - All compute happens in the Triton kernel (no torch.nn.functional or other PyTorch compute ops).
    - Wrapper only validates and launches the kernel.
    """
    assert isinstance(x, torch.Tensor) and isinstance(src, torch.Tensor), "Inputs must be tensors."
    assert x.is_cuda and src.is_cuda, "Both tensors must be on CUDA device."
    assert src.numel() == 1, "src must be a 0-d (scalar) tensor."
    # In-place semantics require direct writes to x's storage; we thus require contiguous memory.
    # Extending to non-contiguous views would need stride-aware indexing logic.
    assert x.is_contiguous(), "Only contiguous tensors are supported in this reference kernel."

    n_elements = x.numel()
    if n_elements == 0:
        return x

    # Choose a reasonable block size (power of 2) as per guidelines.
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fill_tensor_kernel[grid](x, src, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Return the same tensor to match in-place aten.fill_.Tensor semantics
    return x