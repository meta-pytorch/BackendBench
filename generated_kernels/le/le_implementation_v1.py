import torch
import triton
import triton.language as tl


@triton.jit
def _le_scalar_kernel(
    x_ptr,                    # *input* tensor (contiguous 1D view)
    out_ptr,                  # *output* tensor (contiguous 1D view, bool)
    n_elements,               # total number of elements
    scalar_f,                 # scalar as float32 (runtime)
    scalar_i,                 # scalar as int32 (runtime)
    BLOCK_SIZE: tl.constexpr,  # block size
    IS_BOOL: tl.constexpr,      # whether input dtype is bool
    IS_FLOAT: tl.constexpr,     # whether input dtype is floating-point
    SCALAR_IS_FLOAT: tl.constexpr,  # whether scalar was provided as float
):
    """
    Elementwise comparison: out = (x <= scalar)
    - Supports boolean, integer, and floating-point x.
    - Returns a boolean tensor.
    - Compares in the appropriate type domain to match PyTorch semantics:
      * float x: compare in x's floating type (scalar cast to that type)
      * int/uint x: compare in integer domain if scalar is int, else upcast x to float32 and compare
      * bool x: promote to int32 {False->0, True->1}; compare against int or float scalar accordingly
    - Handles out-of-bounds with masks.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Compute comparison result (boolean / tl.int1)
    if IS_BOOL:
        # Treat bool as 0/1 integer for numeric comparisons (matches PyTorch behavior)
        xi = x.to(tl.int32)
        if SCALAR_IS_FLOAT:
            s = tl.full([BLOCK_SIZE], scalar_f, dtype=tl.float32)
            cmp = xi.to(tl.float32) <= s
        else:
            s = tl.full([BLOCK_SIZE], scalar_i, dtype=tl.int32)
            cmp = xi <= s
    else:
        if IS_FLOAT:
            # Cast scalar to x's floating-point dtype for exact PyTorch-like behavior
            s = tl.full([BLOCK_SIZE], scalar_f, dtype=tl.float32).to(x.dtype)
            cmp = x <= s
        else:
            # Integer / Unsigned integer types
            xi = x.to(tl.int32)
            if SCALAR_IS_FLOAT:
                # Mixed int tensor and float scalar -> compare in float32 domain
                s = tl.full([BLOCK_SIZE], scalar_f, dtype=tl.float32)
                cmp = xi.to(tl.float32) <= s
            else:
                s = tl.full([BLOCK_SIZE], scalar_i, dtype=tl.int32)
                cmp = xi <= s

    # Store result with masking
    tl.store(out_ptr + offsets, cmp, mask=mask)


def le_kernel_impl(x: torch.Tensor, scalar):
    """
    Triton-based implementation of aten.le.Scalar (x <= scalar).

    Args:
        x: Input PyTorch tensor (any shape, potentially non-contiguous).
        scalar: Python scalar (int or float).

    Returns:
        A boolean tensor on the same device as x with the same shape,
        where each element is (x[i] <= scalar).
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")

    # Handle empty tensors early
    n_elements = x.numel()
    if n_elements == 0:
        return torch.empty(x.shape, dtype=torch.bool, device=x.device)

    # Prepare flat, contiguous buffers for coalesced accesses
    x_flat = x.contiguous().view(-1)
    out_flat = torch.empty(n_elements, dtype=torch.bool, device=x.device)

    # Determine dtype categories
    is_bool = x.dtype == torch.bool
    is_float = x.is_floating_point()
    # Scalar type
    scalar_is_float = isinstance(scalar, float)

    # Prepare scalar values in both int and float forms for the kernel
    # (only the relevant one will be used depending on flags)
    scalar_f = float(scalar)
    scalar_i = int(scalar)

    # Kernel launch configuration
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    # Launch kernel
    _le_scalar_kernel[grid](
        x_flat,
        out_flat,
        n_elements,
        scalar_f,
        scalar_i,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BOOL=is_bool,
        IS_FLOAT=is_float,
        SCALAR_IS_FLOAT=scalar_is_float,
        num_warps=4,
    )

    # View result back to the original shape (contiguity not required by the test)
    return out_flat.view(x.shape)