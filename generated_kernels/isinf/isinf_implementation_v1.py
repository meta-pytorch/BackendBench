import torch
import triton
import triton.language as tl


@triton.jit
def _isinf_kernel(x_ptr, out_u8_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise isinf kernel.
    Loads elements from x_ptr, checks for +inf or -inf, and writes 1 (True) or 0 (False) to out_u8_ptr.
    - x_ptr: pointer to input tensor (float16/bfloat16/float32)
    - out_u8_ptr: pointer to output tensor (uint8), where 1 means True and 0 means False
    - n_elements: total number of elements to process
    - BLOCK_SIZE: compile-time constant controlling number of elements per program
    """
    # 1D indexing: each program handles a block of BLOCK_SIZE elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to guard out-of-bounds
    mask = offsets < n_elements

    # Load inputs with masking; out-of-bounds lanes get a neutral value (0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute isinf: True if x == +inf or x == -inf
    is_pos_inf = x == float("inf")
    is_neg_inf = x == float("-inf")
    is_inf = is_pos_inf | is_neg_inf

    # Store result as uint8 (1 for True, 0 for False)
    is_inf_u8 = is_inf.to(tl.uint8)
    tl.store(out_u8_ptr + offsets, is_inf_u8, mask=mask)


def isinf_kernel_impl(tensor: torch.Tensor) -> torch.Tensor:
    """
    Triton-based implementation of aten.isinf.default (torch.isinf).

    This wrapper:
    - Validates inputs and allocates an output tensor of dtype torch.bool
    - Ensures coalesced memory access by using a contiguous view of the input
    - Launches a Triton kernel to compute isinf elementwise
    - Returns a boolean tensor of the same shape on the same device

    Notes:
    - Handles empty tensors.
    - Handles non-contiguous inputs by operating on a contiguous copy (values preserved).
    - Supports float16 and bfloat16 (and will also work with float32 if provided).
    """
    if not tensor.is_cuda:
        raise RuntimeError("kernel_function requires a CUDA tensor.")

    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype {tensor.dtype}. Supported: torch.float16, torch.bfloat16, torch.float32."
        )

    n_elements = tensor.numel()
    device = tensor.device

    # Output must be boolean with the same shape
    out_bool = torch.empty(tensor.shape, dtype=torch.bool, device=device)

    # Early exit for empty tensors
    if n_elements == 0:
        return out_bool

    # For optimal memory access (coalescing), use a contiguous view of the input.
    # This does not change values; it only ensures linear addressing in the kernel.
    x_contig = tensor.contiguous()

    # We store kernel results as uint8 (0/1), then cast to bool.
    out_u8 = torch.empty(tensor.shape, dtype=torch.uint8, device=device)

    # Kernel launch configuration
    BLOCK_SIZE = 1024  # power-of-two for performance
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    _isinf_kernel[grid](
        x_contig, out_u8, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )

    # Convert to bool and return
    out_bool.copy_(out_u8.to(dtype=torch.bool))
    return out_bool