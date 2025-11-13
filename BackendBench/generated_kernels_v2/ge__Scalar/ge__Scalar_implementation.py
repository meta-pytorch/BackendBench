# kernel.py
"""
Triton implementation of aten.pow.Scalar for the (Scalar base, Tensor exponent) variant.

What this implements:
- Given a scalar base `a` and a tensor exponent `x`, compute y = a ** x elementwise on GPU.
- The output tensor has the same shape and dtype as the exponent tensor.

Fusion notes:
- This operator is inherently a single elementwise stage (pow of a scalar and a tensor).
- There are no obvious follow-up stages in the provided test to fuse with.
- We therefore implement a single-pass, elementwise Triton kernel: Load -> Compute -> Store.

Runtime rules satisfied:
- All math is done in Triton kernel using tl.load/tl.store and tl.math operations.
- Python wrapper only validates inputs, allocates output, and launches the kernel.
- No torch.nn, torch.nn.functional, or PyTorch compute ops are used in the execution path.

Edge cases:
- For base <= 0 and non-integer exponents, results will follow IEEE rules (NaNs), consistent with PyTorch.
- For extreme values, overflow/underflow may occur similarly to PyTorch behavior.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pow_scalar_tensor_kernel(exp_ptr, out_ptr, n_elements, base_scalar,  #
                              BLOCK_SIZE: tl.constexpr):
    """
    Compute y[i] = base_scalar ** exp[i] for i in [0, n_elements).
    - Loads exponent in fp32 for better numeric stability, computes in fp32, and casts back to out dtype on store.
    - Uses exp2 and log2 for better performance and numerical behavior.

    Args:
        exp_ptr: pointer to exponent tensor (any floating dtype)
        out_ptr: pointer to output tensor
        n_elements: total number of elements
        base_scalar: scalar base (Python scalar passed to kernel)
        BLOCK_SIZE: compile-time constant block size
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load exponents and upcast to fp32 for compute
    exp_vals = tl.load(exp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute in fp32: a**x = 2**(x * log2(a))
    # base_scalar is a kernel scalar argument; compute log2 once per program.
    log2_base = tl.math.log2(base_scalar)
    y = tl.math.exp2(exp_vals * log2_base)

    # Cast to output dtype and store
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def ge__Scalar_kernel_impl(base_scalar, exponent_tensor):
    """
    Wrapper for aten.pow.Scalar (Scalar base, Tensor exponent).

    Args:
        base_scalar: Python scalar (int/float) acting as the base of the power.
        exponent_tensor: torch.Tensor on CUDA, exponent values. Output shape/dtype matches this tensor.

    Returns:
        torch.Tensor with y = base_scalar ** exponent_tensor elementwise.

    Notes:
        - No fusion opportunities exist in the provided test case beyond the single elementwise operation.
        - All arithmetic is executed inside the Triton kernel.
    """
    # Validate inputs
    assert isinstance(exponent_tensor, torch.Tensor), "exponent_tensor must be a torch.Tensor"
    assert exponent_tensor.is_cuda, "exponent_tensor must be on CUDA device"
    assert exponent_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64), \
        "Only floating exponent dtypes are supported (f16, bf16, f32, f64)."
    # We compute in fp32; if input is f64, result will be cast back to f64 on store.
    # For simplicity we will load as fp32, compute, then cast to output dtype.

    # Allocate output
    out = torch.empty_like(exponent_tensor)

    # Compute total number of elements
    n_elements = exponent_tensor.numel()
    if n_elements == 0:
        return out

    # Choose block size (power-of-two as per guidelines)
    BLOCK_SIZE = 1024

    # Grid: 1D over flattened elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch Triton kernel
    _pow_scalar_tensor_kernel[grid](
        exponent_tensor, out, n_elements, float(base_scalar),  #
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


if __name__ == "__main__":
    # Simple manual check (optional)
    x = torch.randn(512, device="cuda", dtype=torch.float32)
    a = 10000
    y_ref = torch.pow(a, x)
    y_triton = kernel_function(a, x)
    torch.testing.assert_close(y_ref, y_triton, rtol=1e-2, atol=1e-2)
    print("Manual check passed.")