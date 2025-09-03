import torch
import triton
import triton.language as tl


@triton.jit
def _isnan_kernel_strided(
    in_ptr,                         # pointer to input tensor data (float/complex-real view)
    out_ptr,                        # pointer to output tensor data (bool)
    N,                              # total number of logical elements to process (not counting complex's trailing 2)
    S0, S1, S2, S3, S4, S5,         # sizes for up to 6 dimensions (unused dims should be 1)
    ST0, ST1, ST2, ST3, ST4, ST5,   # strides (in element units of in_ptr's dtype) for up to 6 dims
    STRIDE_LAST,                    # stride for the trailing complex component axis (only used when IS_COMPLEX=1)
    IS_COMPLEX: tl.constexpr,       # whether input represents complex values via real view and needs two loads
    NDIM: tl.constexpr,             # number of logical dimensions in the original input tensor (<= 6)
    BLOCK_SIZE: tl.constexpr,       # block size for the kernel
):
    """
    Generic strided 'isnan' kernel.
    - Supports up to 6 dimensions for the original tensor.
    - For complex inputs, pass a real view pointer and strides for original dims and STRIDE_LAST for the 2-component axis.
    - For real inputs, STRIDE_LAST is ignored and IS_COMPLEX=0.

    Addressing:
      - We convert a 1D linear index [0, N) into ND indices via repeated div/mod by sizes.
      - Then compute the input element offset using the provided strides.
      - For complex: load real and imag using STRIDE_LAST and OR their isnan results.
      - Store bool result into a contiguous output buffer at the linear location.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Cast to 64-bit to avoid overflow for large tensors
    offs_i64 = offs.to(tl.int64)

    # Prepare sizes and strides as int64
    s0 = tl.full([BLOCK_SIZE], S0, dtype=tl.int64)
    s1 = tl.full([BLOCK_SIZE], S1, dtype=tl.int64)
    s2 = tl.full([BLOCK_SIZE], S2, dtype=tl.int64)
    s3 = tl.full([BLOCK_SIZE], S3, dtype=tl.int64)
    s4 = tl.full([BLOCK_SIZE], S4, dtype=tl.int64)
    s5 = tl.full([BLOCK_SIZE], S5, dtype=tl.int64)

    st0 = tl.full([BLOCK_SIZE], ST0, dtype=tl.int64)
    st1 = tl.full([BLOCK_SIZE], ST1, dtype=tl.int64)
    st2 = tl.full([BLOCK_SIZE], ST2, dtype=tl.int64)
    st3 = tl.full([BLOCK_SIZE], ST3, dtype=tl.int64)
    st4 = tl.full([BLOCK_SIZE], ST4, dtype=tl.int64)
    st5 = tl.full([BLOCK_SIZE], ST5, dtype=tl.int64)

    # Compute multi-dimensional index and the corresponding strided offset
    # We extract indices from the last dimension to the first: idiv //= size_d and imod = idiv % size_d
    idiv = offs_i64
    offset_elems = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    if NDIM >= 6:
        i5 = idiv % s5
        offset_elems += i5 * st5
        idiv = idiv // s5
    if NDIM >= 5:
        i4 = idiv % s4
        offset_elems += i4 * st4
        idiv = idiv // s4
    if NDIM >= 4:
        i3 = idiv % s3
        offset_elems += i3 * st3
        idiv = idiv // s3
    if NDIM >= 3:
        i2 = idiv % s2
        offset_elems += i2 * st2
        idiv = idiv // s2
    if NDIM >= 2:
        i1 = idiv % s1
        offset_elems += i1 * st1
        idiv = idiv // s1
    if NDIM >= 1:
        i0 = idiv % s0
        offset_elems += i0 * st0
        # idiv //= s0  # not needed further

    # Base pointers advanced by element offsets
    in_offsets = offset_elems
    out_offsets = offs_i64

    if IS_COMPLEX:
        stride_last = tl.full([BLOCK_SIZE], STRIDE_LAST, dtype=tl.int64)
        # load real and imag components
        real_vals = tl.load(in_ptr + in_offsets, mask=mask, other=0)
        imag_vals = tl.load(in_ptr + in_offsets + stride_last, mask=mask, other=0)
        res = (real_vals != real_vals) | (imag_vals != imag_vals)
    else:
        vals = tl.load(in_ptr + in_offsets, mask=mask, other=0)
        res = vals != vals

    tl.store(out_ptr + out_offsets, res, mask=mask)


def _compute_sizes_strides(t: torch.Tensor, max_dims=6):
    """
    Returns:
      sizes: list[int] length <= max_dims
      strides: list[int] length <= max_dims, in elements (not bytes)
      ndim: int
    Pads with 1 for sizes and 0 for strides for unused dims to match max_dims.
    """
    ndim = t.dim()
    assert ndim <= max_dims, f"Tensor with ndim={ndim} exceeds supported max_dims={max_dims}"

    sizes = list(t.shape)
    strides_elems = list(t.stride())

    # Pad up to max_dims with neutral values
    while len(sizes) < max_dims:
        sizes.append(1)
    while len(strides_elems) < max_dims:
        strides_elems.append(0)

    return sizes, strides_elems, ndim


def isnan_kernel_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-based implementation of torch.isnan.

    Args:
      x: Input tensor (supports floating and complex types; integers/bools will return all False).

    Returns:
      A torch.bool tensor of the same shape and device, where each element indicates whether
      the corresponding element in x is NaN. For complex inputs, True if real or imaginary is NaN.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    device = x.device

    # Handle empty tensors early to avoid 0-grid launch
    numel = x.numel()
    out = torch.empty(x.shape, dtype=torch.bool, device=device)
    if numel == 0:
        return out

    # Decide real/complex handling
    is_complex = x.is_complex()
    if is_complex:
        # Real view presents last dimension of size 2 with appropriate strides
        xr = torch.view_as_real(x)
        # Compute sizes/strides for original dims only (exclude the appended 2)
        sizes, strides, ndim = _compute_sizes_strides(x, max_dims=6)
        # The real view is of dtype float32 for complex64, float64 for complex128
        input_ptr = xr
        stride_last = xr.stride(-1)  # typically 1
    else:
        sizes, strides, ndim = _compute_sizes_strides(x, max_dims=6)
        input_ptr = x
        stride_last = 0  # unused

    # Output is contiguous boolean; we write linearly with offsets
    # Kernel launch configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    # Choose Triton in_ptr dtype through the tensor we pass
    # For complex: pass the real view tensor pointer
    _isnan_kernel_strided[grid](
        input_ptr,                         # in_ptr
        out,                               # out_ptr (bool)
        numel,                             # N
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5],
        strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
        stride_last,
        IS_COMPLEX=1 if is_complex else 0,
        NDIM=ndim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# Optional: simple manual test
if __name__ == "__main__":
    if torch.cuda.is_available():
        for dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]:
            x = torch.randn((33, 65, 129), dtype=dtype, device="cuda")
            # Inject special values
            if x.numel() > 0:
                x.view(-1)[0] = float("nan")
                x.view(-1)[-1] = float("nan")
                x.view(-1)[1] = float("inf")
                x.view(-1)[2] = float("-inf")
            y_ref = torch.isnan(x)
            y = kernel_function(x)
            assert torch.equal(y, y_ref), f"Mismatch for dtype={dtype}"
        # Non-contiguous
        base = torch.randn((32, 64, 130), dtype=torch.bfloat16, device="cuda")
        base.view(-1)[0] = float("nan")
        x_nc = base[:, ::2, 1::2]
        y_ref = torch.isnan(x_nc)
        y = kernel_function(x_nc)
        assert torch.equal(y, y_ref), "Mismatch for non-contiguous case"
        print("Quick self-test passed")