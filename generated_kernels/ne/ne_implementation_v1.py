import torch
import triton
import triton.language as tl


@triton.jit
def _ne_scalar_kernel(x_ptr, out_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise 'not equal to scalar' for non-complex tensors.

    Args:
        x_ptr: Pointer to input tensor (any supported dtype except complex).
        out_ptr: Pointer to output tensor (torch.bool).
        n_elements: Total number of elements in input/output.
        scalar: The scalar to compare against (runtime scalar, cast by Triton as needed).
        BLOCK_SIZE: Number of elements per program.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Compare against scalar
    neq = x != scalar

    # Store result
    tl.store(out_ptr + offsets, neq, mask=mask)


@triton.jit
def _ne_scalar_complex_kernel(x_ri_ptr, out_ptr, n_elements, scalar_real, scalar_imag, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise 'not equal to scalar' for complex tensors.

    The input is expected as a contiguous real-imag view: last dimension size 2,
    where the memory layout is [..., 2] with strides (..., 1). We treat it as a
    flat array of length 2 * n_elements and access pairs (real, imag) at indices
    (2*i, 2*i+1).

    Args:
        x_ri_ptr: Pointer to the real-imag view data (float32 for complex64, float64 for complex128).
        out_ptr: Pointer to output tensor (torch.bool).
        n_elements: Number of complex elements.
        scalar_real: Real part of scalar to compare.
        scalar_imag: Imag part of scalar to compare.
        BLOCK_SIZE: Number of elements per program.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)  # indices of complex elements
    mask = idx < n_elements

    base = idx * 2
    # Load real and imag parts
    r = tl.load(x_ri_ptr + base + 0, mask=mask, other=0.0)
    i = tl.load(x_ri_ptr + base + 1, mask=mask, other=0.0)

    # Compare parts: (r != sr) | (i != si)
    neq = (r != scalar_real) | (i != scalar_imag)

    tl.store(out_ptr + idx, neq, mask=mask)


def ne_kernel_impl(tensor: torch.Tensor, scalar):
    """
    Triton-based implementation of aten.ne.Scalar: elementwise tensor != scalar.

    - Supports bool, integer, floating, and complex dtypes.
    - Returns a boolean tensor with the same shape as the input.
    - Works for arbitrary shapes; non-contiguous inputs are handled by making a contiguous copy.
    - All computation happens inside Triton kernels.

    Args:
        tensor: Input PyTorch tensor on CUDA.
        scalar: Python scalar (bool/int/float/complex) to compare against.

    Returns:
        out: torch.Tensor of dtype torch.bool with the same shape as `tensor`.
    """
    if not tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA.")
    device = tensor.device
    n_elements = tensor.numel()

    # Fast path for empty tensors
    if n_elements == 0:
        return torch.empty(tensor.shape, dtype=torch.bool, device=device)

    # Always use contiguous inputs for predictable, coalesced memory access in the kernel
    # This preserves values and shape semantics, though output layout will be contiguous.
    dtype = tensor.dtype

    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    # Prepare output
    out = torch.empty(tensor.shape, dtype=torch.bool, device=device)

    # Complex path: handle by viewing as real-imag pairs and comparing both parts
    if dtype.is_complex:
        # Ensure contiguous complex storage
        x_contig = tensor.contiguous()
        # Real-imag view; for complex64 -> float32, complex128 -> float64
        x_ri = torch.view_as_real(x_contig)  # shape (..., 2), contiguous if x_contig is
        # Convert scalar to complex: cast semantics like PyTorch (real number -> complex(real, 0))
        if isinstance(scalar, complex):
            s_real = float(scalar.real)
            s_imag = float(scalar.imag)
        else:
            # Python bool/int/float to complex with zero imaginary
            s_real = float(scalar)
            s_imag = 0.0

        _ne_scalar_complex_kernel[grid](x_ri, out, n_elements, s_real, s_imag, BLOCK_SIZE=BLOCK_SIZE)
        return out

    # Non-complex path
    x_contig = tensor.contiguous()

    # Cast scalar to an appropriate Python type to mirror PyTorch casting semantics
    # - bool tensor: bool(other)
    # - integer tensors: int(other)
    # - floating tensors (fp16/bf16/fp32/fp64): float(other)
    if dtype == torch.bool:
        s_val = bool(scalar)
    elif dtype.is_floating_point:
        s_val = float(scalar)
    else:
        # Integer-like, includes signed/unsigned integer types
        s_val = int(scalar)

    _ne_scalar_kernel[grid](x_contig, out, n_elements, s_val, BLOCK_SIZE=BLOCK_SIZE)

    return out