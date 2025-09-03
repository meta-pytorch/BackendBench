import torch
import triton
import triton.language as tl


def _normalize_dims(dim, ndim):
    """Normalize dim argument to a sorted list of unique, positive dims."""
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    # Normalize negatives and deduplicate
    norm = []
    seen = set()
    for d in dims:
        if d < 0:
            d += ndim
        if d < 0 or d >= ndim:
            raise ValueError(f"dim {d} out of range for tensor with {ndim} dims")
        if d not in seen:
            norm.append(d)
            seen.add(d)
    # Sort to keep original order of dimensions as in the tensor layout
    norm.sort()
    return norm


def _suffix_cumprod(sizes):
    """Return suffix cumulative products for shape -> used to decode linear index to multi-index.
    cp[i] = product of sizes[i+1:]; cp[last] = 1
    """
    cp = [1] * len(sizes)
    p = 1
    for i in range(len(sizes) - 1, -1, -1):
        cp[i] = p
        p *= sizes[i]
    return cp


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R': 1024}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def _std_corr_kernel(
    x_ptr,                    # *T, input tensor
    out_ptr,                  # *T, output buffer (1D, length OUT_NUMEL)
    out_sizes_ptr,            # *i32, sizes of outer (non-reduced) dims in original order
    out_strides_ptr,          # *i64, strides (in elements) of outer dims in original order
    red_sizes_ptr,            # *i32, sizes of reduced dims in original order
    red_strides_ptr,          # *i64, strides (in elements) of reduced dims in original order
    red_cprods_ptr,           # *i64, suffix cumulative products for reduced dims
    OUT_RANK: tl.constexpr,   # number of outer dims
    RED_RANK: tl.constexpr,   # number of reduced dims
    N,                        # total number of elements reduced over (int32)
    correction,               # correction (int32), e.g., 0 or 1
    BLOCK_R: tl.constexpr,    # reduction tile size
):
    # One program per output element (outer index)
    pid = tl.program_id(0).to(tl.int64)

    # Compute base offset (in elements) for this output coordinate within x, iterating outer dims
    # We decode pid into multi-index over outer dims in reverse order (least significant last)
    base_off = tl.zeros([1], dtype=tl.int64)
    tmp = pid
    # Loop over outer dims in reverse order to extract indices
    for i in range(OUT_RANK):
        idx = (OUT_RANK - 1) - i
        size_i = tl.load(out_sizes_ptr + idx).to(tl.int64)
        stride_i = tl.load(out_strides_ptr + idx)  # already int64
        # idx along this dimension
        dim_idx = tmp % size_i
        tmp = tmp // size_i
        base_off += dim_idx * stride_i

    # Accumulators in input dtype (bf16/fp16 as required by the test)
    dtype = x_ptr.dtype.element_ty
    sum_x = tl.zeros([1], dtype=dtype)
    sum_x2 = tl.zeros([1], dtype=dtype)

    # Reduction over flattened reduced-dims linear index j in [0, N)
    # We build gather offsets for each tile using radix decomposition with suffix cprods.
    for r_start in tl.range(0, N, BLOCK_R):
        j = r_start + tl.arange(0, BLOCK_R)
        mask = j < N

        # Compute offsets within the reduced subspace
        off_r = tl.zeros([BLOCK_R], dtype=tl.int64)
        # For each reduced dim k, add its contribution
        for k in range(RED_RANK):
            size_k = tl.load(red_sizes_ptr + k).to(tl.int64)
            cp_k = tl.load(red_cprods_ptr + k).to(tl.int64)
            stride_k = tl.load(red_strides_ptr + k)  # int64
            idx_k = (j.to(tl.int64) // cp_k) % size_k
            off_r += idx_k * stride_k

        # Gather load
        ptrs = x_ptr + (base_off + off_r)
        vals = tl.load(ptrs, mask=mask, other=0).to(dtype)

        # Accumulate sum and sum of squares
        sum_x += tl.sum(vals, axis=0)
        sum_x2 += tl.sum(vals * vals, axis=0)

    # Compute variance with correction, then std
    # numerator = sum((x - mean)^2) = sum_x2 - sum_x^2 / N
    Nf = tl.full([1], N, dtype=dtype)
    num = sum_x2 - (sum_x * sum_x) / Nf

    # denom = N - correction
    denom_i32 = N - correction
    # Handle denom <= 0 -> NaN
    zero = tl.zeros([1], dtype=dtype)
    nan_val = zero / zero  # NaN in any float dtype

    # For valid denom, compute var = num / denom, clamp to >= 0, std = sqrt(var)
    denf = tl.full([1], denom_i32, dtype=dtype)
    var = num / denf
    # clamp small negatives to zero due to rounding in low precision
    var = tl.where(var < zero, zero, var)
    std = tl.sqrt(var)

    # Select NaN when denom <= 0
    cond_nan = denom_i32 <= 0
    out_val = tl.where(cond_nan, nan_val, std)

    # Store into 1D output
    offs_out = pid + tl.arange(0, 1)
    tl.store(out_ptr + offs_out, out_val)


def std_kernel_impl(x: torch.Tensor, dim=None, correction: int = 1, keepdim: bool = False) -> torch.Tensor:
    """
    Compute standard deviation with correction over specified dimensions using a Triton kernel.
    Functionally mirrors torch.ops.aten.std.correction (operation name: std).

    Args:
        x: Input tensor (CUDA). Tested with bfloat16 and float16.
        dim: None, int, or sequence of ints specifying reduction dims.
        correction: Integer correction (0 -> population, 1 -> unbiased).
        keepdim: Whether to retain reduced dims as size-1.

    Returns:
        Tensor containing standard deviation values with the same dtype as x and shapes per PyTorch semantics.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if x.dtype not in (torch.bfloat16, torch.float16):
        # You can extend support; tests focus on bf16/f16
        raise TypeError(f"Unsupported dtype {x.dtype}. Supported: torch.bfloat16, torch.float16.")

    ndim = x.ndim
    dims = _normalize_dims(dim, ndim)

    # If no reduction dims, just return zeros with same shape? PyTorch: std over empty dims returns same tensor?
    # Not needed for tests; still handle gracefully by following PyTorch: reducing over empty dims returns std along no axes -> result equals input.
    if len(dims) == 0:
        # torch.ops.aten.std.correction with an empty dim reduces nothing -> identical to x
        return x.clone()

    # Build lists of outer (non-reduced) and reduced dims in original order
    red_set = set(dims)
    outer_dims = [d for d in range(ndim) if d not in red_set]
    red_dims = dims  # already sorted ascending (original order)

    # Compute final output shape
    if keepdim:
        out_shape = [1 if i in red_set else x.shape[i] for i in range(ndim)]
    else:
        out_shape = [x.shape[i] for i in outer_dims]

    # Prepare sizes and strides arrays for outer and reduced dims
    x_sizes = list(x.shape)
    x_strides = list(x.stride())  # strides are in elements already

    out_sizes = [x_sizes[d] for d in outer_dims]
    out_strides = [x_strides[d] for d in outer_dims]

    red_sizes = [x_sizes[d] for d in red_dims]
    red_strides = [x_strides[d] for d in red_dims]

    OUT_RANK = len(out_sizes)
    RED_RANK = len(red_sizes)

    # Total elements being reduced over (constant across all outputs)
    N = 1
    for s in red_sizes:
        N *= int(s)

    # Allocate a 1D output buffer for OUT_NUMEL elements
    OUT_NUMEL = 1
    for s in out_sizes:
        OUT_NUMEL *= int(s)
    # Even if OUT_RANK == 0 => OUT_NUMEL == 1
    out_buf = torch.empty((OUT_NUMEL,), device=x.device, dtype=x.dtype)

    # Create device arrays (use at least 1-length placeholders if rank is 0 to avoid null pointers)
    device = x.device
    if OUT_RANK > 0:
        out_sizes_dev = torch.tensor(out_sizes, dtype=torch.int32, device=device)
        out_strides_dev = torch.tensor(out_strides, dtype=torch.int64, device=device)
    else:
        out_sizes_dev = torch.empty(1, dtype=torch.int32, device=device)
        out_strides_dev = torch.empty(1, dtype=torch.int64, device=device)

    if RED_RANK > 0:
        red_sizes_dev = torch.tensor(red_sizes, dtype=torch.int32, device=device)
        red_strides_dev = torch.tensor(red_strides, dtype=torch.int64, device=device)
        red_cprods = _suffix_cumprod(red_sizes)
        red_cprods_dev = torch.tensor(red_cprods, dtype=torch.int64, device=device)
    else:
        # Shouldn't happen in our tests, but keep safe placeholders
        red_sizes_dev = torch.empty(1, dtype=torch.int32, device=device)
        red_strides_dev = torch.empty(1, dtype=torch.int64, device=device)
        red_cprods_dev = torch.empty(1, dtype=torch.int64, device=device)

    # Launch grid: one program per output element
    grid = (OUT_NUMEL,)

    # Launch kernel
    _std_corr_kernel[grid](
        x, out_buf,
        out_sizes_dev, out_strides_dev,
        red_sizes_dev, red_strides_dev, red_cprods_dev,
        OUT_RANK=OUT_RANK,
        RED_RANK=RED_RANK,
        N=N,
        correction=int(correction),
    )

    # Reshape to the expected output shape
    result = out_buf.reshape(out_shape)

    return result