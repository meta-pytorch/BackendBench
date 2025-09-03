import torch
import triton
import triton.language as tl


@triton.jit
def _masked_fill_kernel(
    x_ptr,               # *T: input tensor
    mask_ptr,            # *bool: mask tensor (broadcastable to x)
    out_ptr,             # *T: output tensor (same dtype/shape/strides as x)
    shape_ptr,           # *int64: tensor of length NDIMS with sizes of x
    x_strides_ptr,       # *int64: tensor of length NDIMS with strides of x (in elements)
    m_strides_ptr,       # *int64: tensor of length NDIMS with strides of mask aligned to x (in elements, 0 for broadcasted dims)
    n_elements,          # int32: total number of elements in x
    value_f32,           # float32: scalar value as float32 (used for floating dtypes)
    value_i32,           # int32: scalar value as int32 (used for integer dtypes)
    value_bi32,          # int32: scalar value as 0/1 (used for bool dtype)
    NDIMS: tl.constexpr,       # number of dimensions
    BLOCK_SIZE: tl.constexpr,  # tile size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    # Compute multi-dimensional indices and resulting memory offsets
    # Using row-major (last dimension fastest) index decomposition.
    rem = offs
    x_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    m_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    for d in range(NDIMS - 1, -1, -1):
        size_d = tl.load(shape_ptr + d).to(tl.int32)
        idx_d = rem % size_d
        rem = rem // size_d
        xs_d = tl.load(x_strides_ptr + d).to(tl.int64)
        ms_d = tl.load(m_strides_ptr + d).to(tl.int64)
        x_off += idx_d.to(tl.int64) * xs_d
        m_off += idx_d.to(tl.int64) * ms_d

    # Load input values
    x_vals = tl.load(x_ptr + x_off, mask=in_bounds, other=0)

    # Load mask; convert to boolean (handles tl.int1 or integer types)
    m_raw = tl.load(mask_ptr + m_off, mask=in_bounds, other=0)
    m_bool = m_raw != 0

    # Initialize output with input values
    tl.store(out_ptr + x_off, x_vals, mask=in_bounds)

    # Prepare scalar value vector in the correct dtype of x
    # Then overwrite masked positions.
    if x_ptr.dtype.element_ty == tl.float16:
        val_vec = tl.full([BLOCK_SIZE], value_f32, dtype=tl.float16)
    elif x_ptr.dtype.element_ty == tl.bfloat16:
        val_vec = tl.full([BLOCK_SIZE], value_f32, dtype=tl.bfloat16)
    elif x_ptr.dtype.element_ty == tl.float32:
        val_vec = tl.full([BLOCK_SIZE], value_f32, dtype=tl.float32)
    elif x_ptr.dtype.element_ty == tl.int32:
        val_vec = tl.full([BLOCK_SIZE], value_i32, dtype=tl.int32)
    elif x_ptr.dtype.element_ty == tl.int1:
        # Triton bool is tl.int1; value_bi32 is 0 or 1
        val_vec = tl.full([BLOCK_SIZE], value_bi32, dtype=tl.int1)
    else:
        # Fallback (not expected in provided tests): try to cast from float32
        val_vec = tl.full([BLOCK_SIZE], value_f32, dtype=x_ptr.dtype.element_ty)

    write_mask = in_bounds & m_bool
    tl.store(out_ptr + x_off, val_vec, mask=write_mask)


def _aligned_mask_strides(x: torch.Tensor, mask: torch.Tensor):
    """
    Compute mask strides aligned to x's dimensions, applying broadcasting rules.
    Returns a list of length x.ndim with stride 0 in broadcasted dimensions.
    """
    nd = x.ndim
    md = mask.ndim
    msizes = [1] * nd
    mstrides = [0] * nd
    # Right-align mask dims with x dims
    for i in range(md):
        msizes[nd - md + i] = mask.shape[i]
        mstrides[nd - md + i] = mask.stride(i)
    # Broadcast dims (size==1) must have stride 0
    for d in range(nd):
        if msizes[d] == 1:
            mstrides[d] = 0
    return mstrides


def masked_fill_kernel_impl(x: torch.Tensor, mask: torch.Tensor, value):
    """
    Triton-based masked_fill implementation.

    Args:
        x: Input tensor (on CUDA). Supports dtypes: bfloat16, float16, int32, bool
        mask: Boolean tensor broadcastable to x
        value: Python scalar (float, int, or bool). Will be cast to x.dtype.

    Returns:
        Tensor of same shape and dtype as x with elements set to `value` where mask is True.
    """
    assert x.is_cuda, "Input x must be on CUDA."
    assert mask.is_cuda, "Mask must be on CUDA."
    # Ensure mask is boolean
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    out = torch.empty_like(x)

    ndims = x.ndim
    n_elements = x.numel()

    # Shape and strides (in elements)
    shape_t = torch.tensor(list(x.shape), device=x.device, dtype=torch.int64)
    x_strides_t = torch.tensor(list(x.stride()), device=x.device, dtype=torch.int64)
    m_strides_list = _aligned_mask_strides(x, mask)
    m_strides_t = torch.tensor(m_strides_list, device=x.device, dtype=torch.int64)

    # Scalar representations for kernel (we pass all forms; kernel picks the one it needs)
    value_f32 = float(value)
    value_i32 = int(value)
    value_bi32 = int(bool(value))

    # Launch configuration
    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _masked_fill_kernel[grid](
        x, mask, out,
        shape_t, x_strides_t, m_strides_t,
        n_elements,
        value_f32, value_i32, value_bi32,
        NDIMS=ndims,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out