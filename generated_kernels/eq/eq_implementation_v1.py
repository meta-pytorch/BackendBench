import triton
import triton.language as tl
import torch


"""
Triton kernel implementation for aten.eq.Scalar (elementwise equality against a scalar).

Features:
- Works for tensors of arbitrary shapes (up to 8 dimensions handled explicitly here)
- Supports non-contiguous tensors and tensors with zero-stride (expanded) dimensions
- Supports dtypes:
  * Integers: int8, int16, int32, int64, uint8
  * Floating point: float16, bfloat16, float32, float64
  * Boolean (handled via byte view)
  * Complex: complex64, complex128 (compares to complex(scalar, 0))
- Produces a boolean (torch.bool) tensor as output, identical to PyTorch's aten.eq.Scalar
- Proper masking for boundary conditions
- Coalesced memory access over a flat, contiguous output layout
- Autotuned block sizes for performance

Notes:
- For float/bfloat16 types, comparisons are performed in float32 for numerical stability.
- For float64, comparisons are performed in float64.
- For integer types, comparisons are performed in int64 (avoids overflow and unifies logic).
- For bool inputs, computation is performed on a uint8 view (0 or 1), while output stays torch.bool.

API:
    kernel_function(tensor, scalar) -> torch.Tensor[bool] of same shape as `tensor`
"""


def _pack_shape_strides(t: torch.Tensor, max_dims: int = 8):
    """
    Pack tensor shape and strides into fixed-length lists of length max_dims.
    Strides are in units of elements (PyTorch's strides already are).
    """
    shape = list(t.shape)
    strides = list(t.stride())
    assert len(shape) <= max_dims, f"Tensor with rank > {max_dims} not supported in this kernel."

    # Left-pad to max_dims with 1s for shapes and 0s for strides (no contribution)
    pad = max_dims - len(shape)
    shape = [1] * pad + shape
    strides = [0] * pad + strides
    return shape, strides


# Autotune configurations for elementwise kernels
_configs = [
    triton.Config({"BLOCK_SIZE": bs}, num_stages=2, num_warps=w)
    for bs in [64, 128, 256, 512, 1024]
    for w in [2, 4, 8]
]


@triton.autotune(configs=_configs, key=["N_ELEMENTS"])
@triton.jit
def _eq_scalar_strided_kernel(
    x_ptr,                  # * pointer to input tensor (any non-complex dtype)
    out_ptr,                # * pointer to output (bool storage, 1 byte per element)
    scalar_f,               # scalar value as float (used for float family)
    scalar_i,               # scalar value as int   (used for integer/bool family)
    N_ELEMENTS,             # total number of elements
    S0, S1, S2, S3, S4, S5, S6, S7,        # shape per dimension (padded to 8D)
    STR0, STR1, STR2, STR3, STR4, STR5, STR6, STR7,  # strides per dimension (in elements)
    IS_FLOAT: tl.constexpr,               # whether x is floating family (fp16/bf16/fp32/fp64)
    USE_FP64: tl.constexpr,               # whether x is float64 (else compare in float32)
    IS_BOOL: tl.constexpr,                # whether x is bool (we pass a uint8 view)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEMENTS

    # Compute multi-dimensional indices from linear index (row-major, last dim fastest)
    idx = offs.to(tl.int64)
    i7 = (idx % S7).to(tl.int64); idx = idx // S7
    i6 = (idx % S6).to(tl.int64); idx = idx // S6
    i5 = (idx % S5).to(tl.int64); idx = idx // S5
    i4 = (idx % S4).to(tl.int64); idx = idx // S4
    i3 = (idx % S3).to(tl.int64); idx = idx // S3
    i2 = (idx % S2).to(tl.int64); idx = idx // S2
    i1 = (idx % S1).to(tl.int64); idx = idx // S1
    i0 = idx.to(tl.int64)

    # Compute input element offsets using strides (in elements)
    off_elems = (
        i0 * STR0
        + i1 * STR1
        + i2 * STR2
        + i3 * STR3
        + i4 * STR4
        + i5 * STR5
        + i6 * STR6
        + i7 * STR7
    )
    x_ptrs = x_ptr + off_elems

    # Load input; "other=0" is safe due to mask
    x = tl.load(x_ptrs, mask=mask, other=0)

    # Broadcast scalar to a vector of appropriate dtype and compare
    if IS_BOOL:
        # Treat input as uint8 (0/1)
        x_u8 = x.to(tl.uint8)
        s_u8 = tl.full([BLOCK_SIZE], scalar_i, dtype=tl.uint8)
        eq = x_u8 == s_u8
    elif IS_FLOAT:
        if USE_FP64:
            x_f = x.to(tl.float64)
            s_f = tl.full([BLOCK_SIZE], scalar_f, dtype=tl.float64)
        else:
            x_f = x.to(tl.float32)
            s_f = tl.full([BLOCK_SIZE], scalar_f, dtype=tl.float32)
        eq = x_f == s_f
    else:
        x_i = x.to(tl.int64)
        s_i = tl.full([BLOCK_SIZE], scalar_i, dtype=tl.int64)
        eq = x_i == s_i

    # Store result as bytes (0/1) into bool storage
    out_vals = eq.to(tl.uint8)
    tl.store(out_ptr + offs, out_vals, mask=mask)


@triton.autotune(configs=_configs, key=["N_ELEMENTS"])
@triton.jit
def _eq_scalar_complex_strided_kernel(
    xr_ptr, xi_ptr,        # pointers to real and imaginary views (float32/64)
    out_ptr,               # pointer to output (bool storage)
    scalar_f,              # scalar as float; compare to complex(scalar, 0)
    N_ELEMENTS,            # total number of elements
    S0, S1, S2, S3, S4, S5, S6, S7,        # shape per dimension
    STR0, STR1, STR2, STR3, STR4, STR5, STR6, STR7,  # strides per dimension (elements of real/imag dtype)
    REAL_IS_FP64: tl.constexpr,            # whether real/imag are float64 (else float32)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEMENTS

    # Compute multi-dimensional indices
    idx = offs.to(tl.int64)
    i7 = (idx % S7).to(tl.int64); idx = idx // S7
    i6 = (idx % S6).to(tl.int64); idx = idx // S6
    i5 = (idx % S5).to(tl.int64); idx = idx // S5
    i4 = (idx % S4).to(tl.int64); idx = idx // S4
    i3 = (idx % S3).to(tl.int64); idx = idx // S3
    i2 = (idx % S2).to(tl.int64); idx = idx // S2
    i1 = (idx % S1).to(tl.int64); idx = idx // S1
    i0 = idx.to(tl.int64)

    off_elems = (
        i0 * STR0
        + i1 * STR1
        + i2 * STR2
        + i3 * STR3
        + i4 * STR4
        + i5 * STR5
        + i6 * STR6
        + i7 * STR7
    )
    xr_ptrs = xr_ptr + off_elems
    xi_ptrs = xi_ptr + off_elems

    xr = tl.load(xr_ptrs, mask=mask, other=0.0)
    xi = tl.load(xi_ptrs, mask=mask, other=0.0)

    if REAL_IS_FP64:
        xr_f = xr.to(tl.float64)
        xi_f = xi.to(tl.float64)
        s_f = tl.full([BLOCK_SIZE], scalar_f, dtype=tl.float64)
        z_f = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float64)
    else:
        xr_f = xr.to(tl.float32)
        xi_f = xi.to(tl.float32)
        s_f = tl.full([BLOCK_SIZE], scalar_f, dtype=tl.float32)
        z_f = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)

    eq = (xr_f == s_f) & (xi_f == z_f)
    out_vals = eq.to(tl.uint8)
    tl.store(out_ptr + offs, out_vals, mask=mask)


def eq_kernel_impl(tensor: torch.Tensor, scalar):
    """
    Wrapper function that launches the Triton kernels.

    Args:
      tensor: input PyTorch tensor on CUDA
      scalar: Python scalar (int/bool/float). For complex tensors, scalar is treated as complex(scalar, 0).

    Returns:
      torch.Tensor: boolean tensor of the same shape as `tensor`, with elementwise results of (tensor == scalar).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("kernel_function expects a torch.Tensor as the first argument.")
    if not tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")

    device = tensor.device
    numel = tensor.numel()

    # Allocate contiguous boolean output
    out = torch.empty(tensor.shape, dtype=torch.bool, device=device)

    # Early exit for empty tensors
    if numel == 0:
        return out

    # Prepare shape and strides (up to 8D)
    # For complex case, we build from real/imag views.
    if tensor.is_complex():
        # Complex path: use real/imag float views
        xr = tensor.real
        xi = tensor.imag
        shape, strides = _pack_shape_strides(xr, max_dims=8)

        # Determine real dtype
        if xr.dtype == torch.float64:
            real_is_fp64 = True
        elif xr.dtype == torch.float32:
            real_is_fp64 = False
        else:
            raise TypeError(f"Unsupported complex real dtype: {xr.dtype}")

        # Kernel launch parameters
        N_ELEMENTS = numel
        grid = lambda META: (triton.cdiv(N_ELEMENTS, META["BLOCK_SIZE"]),)

        _eq_scalar_complex_strided_kernel[grid](
            xr, xi, out,
            float(scalar),
            N_ELEMENTS,
            shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7],
            strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6], strides[7],
            REAL_IS_FP64=real_is_fp64,
        )
        return out

    # Non-complex path
    dt = tensor.dtype

    # For bool, use a uint8 view for robust loads/comparisons in kernel.
    if dt == torch.bool:
        x_view = tensor.view(torch.uint8)
        is_bool = True
        is_float = False
        use_fp64 = False
    else:
        x_view = tensor
        is_bool = False
        # dtype family checks
        if dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            is_float = True
            use_fp64 = dt == torch.float64
        elif dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            is_float = False
            use_fp64 = False
        else:
            raise TypeError(f"Unsupported dtype: {dt}")

    shape, strides = _pack_shape_strides(x_view, max_dims=8)

    N_ELEMENTS = numel
    grid = lambda META: (triton.cdiv(N_ELEMENTS, META["BLOCK_SIZE"]),)

    # Launch kernel
    _eq_scalar_strided_kernel[grid](
        x_view, out,
        float(scalar), int(bool(scalar)) if is_bool else int(scalar),
        N_ELEMENTS,
        shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7],
        strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6], strides[7],
        IS_FLOAT=is_float,
        USE_FP64=use_fp64,
        IS_BOOL=is_bool,
    )
    return out