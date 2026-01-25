import torch
import triton
import triton.language as tl


@triton.jit
def _pow_kernel(inp_ptr, out_ptr, n_elements, scalar,  #
                MODE: tl.constexpr,  #
                BLOCK_SIZE: tl.constexpr):
    """
    Generic elementwise power kernel.

    MODE:
      - 0: scalar_base ^ tensor_exponent  => out[i] = scalar ** inp[i]
      - 1: tensor_base ^ scalar_exponent  => out[i] = inp[i] ** scalar

    Notes:
      - All math happens in the kernel. Wrapper only allocates/launches.
      - Computation is performed in fp32 for numerical stability, cast back on store.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input tensor elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)

    # Compute in float32 regardless of input dtype for better accuracy
    x_f32 = x.to(tl.float32)
    scalar_f32 = tl.full((), scalar, tl.float32)

    # Compute result using exp2/log2 to implement pow:
    # a ** b = exp2(b * log2(a))
    if MODE == 0:
        # scalar base, tensor exponent: out = scalar ** x
        log2_base = tl.math.log2(scalar_f32)
        y = tl.math.exp2(x_f32 * log2_base)
    else:
        # tensor base, scalar exponent: out = x ** scalar
        # This is mathematically undefined for negative bases and non-integer exponents,
        # which matches PyTorch behavior (NaN/Inf). We rely on log2 to propagate NaNs accordingly.
        log2_x = tl.math.log2(x_f32)
        y = tl.math.exp2(log2_x * scalar_f32)

    # Cast to output dtype and store
    out_dtype = out_ptr.dtype.element_ty
    y = y.to(out_dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def pow__Scalar_kernel_impl(arg0, arg1):
    """
    Elementwise power implemented in a single Triton kernel.

    Supports both overloads depending on argument types:
      - scalar base, tensor exponent: kernel_function(base_scalar, exponent_tensor)
      - tensor base, scalar exponent: kernel_function(base_tensor, exponent_scalar)

    Fusion notes:
      - This operator is a single elementwise op (pow) and is fully handled in one pass:
        Load -> Compute (exp2/log2-based pow) -> Store.
      - No additional stages exist to fuse in this test case.

    Runtime behavior:
      - The wrapper validates args, allocates the output, and launches the kernel.
      - All math is computed inside the Triton kernel (no PyTorch compute ops used).
    """
    # Determine overload from arg types
    is_tensor0 = torch.is_tensor(arg0)
    is_tensor1 = torch.is_tensor(arg1)
    is_scalar0 = isinstance(arg0, (int, float))
    is_scalar1 = isinstance(arg1, (int, float))

    if is_scalar0 and is_tensor1:
        # scalar base, tensor exponent
        base_scalar = float(arg0)
        exp = arg1
        assert exp.is_cuda, "Input tensor must be on CUDA device"
        out = torch.empty_like(exp)
        n_elements = out.numel()
        BLOCK_SIZE = 1024

        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _pow_kernel[grid](
            exp, out, n_elements, base_scalar,
            MODE=0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out

    elif is_tensor0 and is_scalar1:
        # tensor base, scalar exponent
        base = arg0
        exponent_scalar = float(arg1)
        assert base.is_cuda, "Input tensor must be on CUDA device"
        out = torch.empty_like(base)
        n_elements = out.numel()
        BLOCK_SIZE = 1024

        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _pow_kernel[grid](
            base, out, n_elements, exponent_scalar,
            MODE=1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out

    else:
        raise TypeError(
            "kernel_function expects (scalar, tensor) or (tensor, scalar). "
            f"Got types: ({type(arg0)}, {type(arg1)})"
        )