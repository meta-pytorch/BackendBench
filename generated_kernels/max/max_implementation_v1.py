import torch
import triton
import triton.language as tl


# ---------------------------
# Triton kernels
# ---------------------------

@triton.jit
def _reduce_input_to_partials(
    x_ptr,                # *input* tensor ptr
    n_elements,           # number of elements in input
    out_vals_ptr,         # per-block partial max values
    out_nan_ptr,          # per-block nan flags (uint8 0/1)
    other_val,            # identity "lowest" value for dtype (e.g., -inf, min int, False)
    BLOCK_SIZE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    """
    First-stage reduction kernel:
    - Each program reduces BLOCK_SIZE contiguous elements into one partial max.
    - For floating types, also records whether any NaN was encountered in the block.
    - For bool, do a max-reduction in integer space (equivalent to logical OR).
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load elements with out-of-bounds masked to the identity "lowest" value
    x = tl.load(x_ptr + offs, mask=mask, other=other_val)

    if IS_FLOAT:
        # NaN detection without tl.isnan: NaN != NaN
        isn = x != x
        # Reduce nan flags via max over uint8 (0/1)
        any_nan = tl.max(isn.to(tl.uint8), axis=0)
        # Replace NaNs with -inf for max computation
        minf = -float("inf")
        x = tl.where(isn, minf, x)
        # Reduce to local max
        local_max = tl.max(x, axis=0)
        # Store outputs
        tl.store(out_vals_ptr + pid, local_max)
        tl.store(out_nan_ptr + pid, any_nan)
    else:
        if IS_BOOL:
            # For bool, cast to int8 to perform reduction safely (max over 0/1)
            xi8 = x.to(tl.int8)
            local_max_i8 = tl.max(xi8, axis=0)
            local_max_bool = local_max_i8 > 0
            tl.store(out_vals_ptr + pid, local_max_bool.to(tl.int1))
        else:
            # Integer path: straightforward max
            local_max = tl.max(x, axis=0)
            tl.store(out_vals_ptr + pid, local_max)
        # No NaN for non-floats
        tl.store(out_nan_ptr + pid, 0)


@triton.jit
def _reduce_partials(
    in_vals_ptr,         # input partial values
    in_nan_ptr,          # input partial nan flags (uint8)
    n_elements,          # number of partials
    out_vals_ptr,        # output partial values
    out_nan_ptr,         # output partial nan flags
    other_val,           # identity "lowest" value for dtype (e.g., -inf, min int, False)
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    """
    Generic reduction kernel for subsequent stages:
    - Reduces the arrays of partial values and partial NaN flags into fewer partials.
    - Works for both float and non-float dtypes because NaN flags are provided as uint8.
    - For bool, perform reduction in integer space and cast back to bool on store.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Reduce values
    vals = tl.load(in_vals_ptr + offs, mask=mask, other=other_val)
    if IS_BOOL:
        vals_i8 = vals.to(tl.int8)
        local_max_i8 = tl.max(vals_i8, axis=0)
        local_max_bool = local_max_i8 > 0
        tl.store(out_vals_ptr + pid, local_max_bool.to(tl.int1))
    else:
        local_max = tl.max(vals, axis=0)
        tl.store(out_vals_ptr + pid, local_max)

    # Reduce NaN flags: use 0 for masked elements; "any" via max over 0/1
    nan_flags = tl.load(in_nan_ptr + offs, mask=mask, other=0)
    local_any_nan = tl.max(nan_flags, axis=0)
    tl.store(out_nan_ptr + pid, local_any_nan)


@triton.jit
def _finalize_kernel(
    in_val_ptr,    # pointer to 1-element tensor containing final value (float/int/bool)
    in_nan_ptr,    # pointer to 1-element uint8 tensor containing final has_nan flag
    out_ptr,       # pointer to 1-element output tensor (same dtype as input)
    IS_FLOAT: tl.constexpr,
):
    """
    Finalize step:
    - If dtype is floating and has_nan flag is set, store NaN; else store the value.
    - For non-float dtypes, just forward the value.
    """
    if IS_FLOAT:
        v = tl.load(in_val_ptr)
        has_nan = tl.load(in_nan_ptr).to(tl.int1)
        nan_v = float("nan")
        out = tl.where(has_nan, nan_v, v)
        tl.store(out_ptr, out)
    else:
        v = tl.load(in_val_ptr)
        tl.store(out_ptr, v)


# ---------------------------
# Python wrapper and helpers
# ---------------------------

def _is_floating_dtype(dtype: torch.dtype) -> bool:
    return dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        getattr(torch, "float8_e5m2", torch.float32),
        getattr(torch, "float8_e4m3fn", torch.float32),
        getattr(torch, "float8_e4m3fnuz", torch.float32),
        getattr(torch, "float8_e5m2fnuz", torch.float32),
    )


def _lowest_value_for_dtype(dtype: torch.dtype):
    """
    Identity/lowest value for max-reduction:
    - float: -inf
    - bool: False
    - unsigned: 0
    - signed: iinfo(dtype).min
    """
    if dtype == torch.bool:
        return False
    if _is_floating_dtype(dtype) or dtype.is_floating_point:
        return float("-inf")
    if dtype == torch.uint8:
        return 0
    try:
        return torch.iinfo(dtype).min
    except Exception:
        return 0


def _launch_first_stage(x_contig: torch.Tensor, block_size: int, num_warps: int):
    """
    Launch the first-stage reduction from input tensor to partials.
    Returns: (vals, nans) partial tensors.
    """
    n_elements = x_contig.numel()
    if n_elements == 0:
        raise RuntimeError("max(): Expected reduction over non-empty tensor")

    num_blocks = triton.cdiv(n_elements, block_size)
    device = x_contig.device
    dtype = x_contig.dtype
    is_float = _is_floating_dtype(dtype)
    is_bool = dtype == torch.bool

    # Output buffers for partials
    partial_vals = torch.empty((num_blocks,), device=device, dtype=dtype)
    partial_nans = torch.empty((num_blocks,), device=device, dtype=torch.uint8)

    other_val = _lowest_value_for_dtype(dtype)

    grid = (num_blocks,)
    _reduce_input_to_partials[grid](
        x_contig, n_elements,
        partial_vals, partial_nans,
        other_val,
        BLOCK_SIZE=block_size,
        IS_FLOAT=is_float,
        IS_BOOL=is_bool,
        num_warps=num_warps,
        num_stages=2,
    )
    return partial_vals, partial_nans


def _launch_next_stage(partial_vals: torch.Tensor, partial_nans: torch.Tensor, block_size: int, num_warps: int):
    """
    Launch a subsequent stage reduction on partials until they fit into a single element.
    Returns: (reduced_vals, reduced_nans)
    """
    assert partial_vals.shape == partial_nans.shape
    n_elements = partial_vals.numel()
    num_blocks = triton.cdiv(n_elements, block_size)

    if n_elements == 1:
        return partial_vals, partial_nans

    device = partial_vals.device
    dtype = partial_vals.dtype
    other_val = _lowest_value_for_dtype(dtype)
    is_bool = dtype == torch.bool

    out_vals = torch.empty((num_blocks,), device=device, dtype=dtype)
    out_nans = torch.empty((num_blocks,), device=device, dtype=torch.uint8)

    grid = (num_blocks,)
    _reduce_partials[grid](
        partial_vals, partial_nans, n_elements,
        out_vals, out_nans,
        other_val,
        BLOCK_SIZE=block_size,
        IS_BOOL=is_bool,
        num_warps=num_warps,
        num_stages=2,
    )
    return out_vals, out_nans


def _finalize_to_scalar(partial_vals: torch.Tensor, partial_nans: torch.Tensor):
    """
    Finalize the result to a 0-dim tensor (scalar) matching input dtype/device.
    """
    assert partial_vals.numel() == 1 and partial_nans.numel() == 1
    device = partial_vals.device
    dtype = partial_vals.dtype
    is_float = _is_floating_dtype(dtype)

    out = torch.empty((), device=device, dtype=dtype)
    _finalize_kernel[(1,)](
        partial_vals, partial_nans, out,
        IS_FLOAT=is_float,
        num_warps=1,
        num_stages=1,
    )
    return out


def max_kernel_impl(x: torch.Tensor):
    """
    High-performance Triton implementation of aten.max.default (torch.max(tensor))
    - Reduces all elements to a single scalar maximum
    - Supports floats (with NaN semantics), integers, and bool
    - Works with non-contiguous tensors via a contiguous copy
    - Raises on empty tensor to match PyTorch behavior
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("kernel_function expects a single input tensor")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    if x.numel() == 0:
        raise RuntimeError("max(): Expected reduction over non-empty tensor")

    # Use contiguous layout for coalesced access
    x_contig = x.contiguous()

    # Tunable parameters
    BLOCK_SIZE = 1024  # power of 2
    NUM_WARPS = 4

    # Stage 1: reduce input to partials
    vals, nans = _launch_first_stage(x_contig, BLOCK_SIZE, NUM_WARPS)

    # Subsequent stages: reduce partials until 1 remains
    while vals.numel() > 1:
        vals, nans = _launch_next_stage(vals, nans, BLOCK_SIZE, NUM_WARPS)

    # Finalize result (handle NaN propagation for floats)
    out = _finalize_to_scalar(vals, nans)
    return out