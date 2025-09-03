import torch
import triton
import triton.language as tl


"""
Triton kernel implementation for in-place fill_ (aten.fill_.Scalar) that supports:
- int64, int32, int16, bool, bfloat16, complex64
- Contiguous and non-contiguous tensors (including negative strides)
- Zero-dim tensors and empty tensors

Key points:
- The actual data writes are performed inside Triton kernels using tl.store.
- We compute strided addresses directly in the kernel using sizes and strides.
- For complex64, we write the real and imaginary parts explicitly as two float32 values.
- The wrapper function kernel_function handles dispatch, grid calculation, and argument setup.
"""

# Reasonable defaults for general kernels
BLOCK_SIZE_DEFAULT = 1024
MAX_RANK_DEFAULT = 8


@triton.jit
def _fill_strided_int64(x_ptr, sizes_ptr, strides_ptr, n_elements,  #
                        BLOCK_SIZE: tl.constexpr, MAX_RANK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    linear = block_start + tl.arange(0, BLOCK_SIZE)
    mask = linear < n_elements
    linear = linear.to(tl.int64)

    # Compute strided offsets from linear indices.
    tmp = linear
    offset_elems = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    # sizes_ptr[d], strides_ptr[d] are expected to be padded up to MAX_RANK:
    # sizes[d] = actual_size or 1; strides[d] = actual_stride or 0
    for d in range(MAX_RANK):
        sz = tl.load(sizes_ptr + d)
        st = tl.load(strides_ptr + d)
        idx_d = tmp % sz
        tmp = tmp // sz
        offset_elems += idx_d * st

    ptrs = x_ptr + offset_elems
    v = tl.full((BLOCK_SIZE,), 0, dtype=tl.int64)  # value placeholder; will be overwritten by argument 'v_in'
    # Triton doesn't support scalar arguments with name override at store, so we pass 'v_in' via pointer arugment? No.
    # Use tl.full with constant (inlined) 'value' argument; set below within wrapper call using keyword.
    # This function definition cannot reference a Python variable directly.
    # We'll pass 'value' as an argument and re-create a bf16/int/float vector from it below.

@triton.jit
def _fill_strided_int64_impl(x_ptr, sizes_ptr, strides_ptr, n_elements, value,  #
                             BLOCK_SIZE: tl.constexpr, MAX_RANK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    linear = block_start + tl.arange(0, BLOCK_SIZE)
    mask = linear < n_elements
    linear = linear.to(tl.int64)

    tmp = linear
    offset_elems = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for d in range(MAX_RANK):
        sz = tl.load(sizes_ptr + d)
        st = tl.load(strides_ptr + d)
        idx_d = tmp % sz
        tmp = tmp // sz
        offset_elems += idx_d * st

    ptrs = x_ptr + offset_elems
    v = tl.full((BLOCK_SIZE,), value, dtype=tl.int64)
    tl.store(ptrs, v, mask=mask)


@triton.jit
def _fill_strided_int32_impl(x_ptr, sizes_ptr, strides_ptr, n_elements, value,  #
                             BLOCK_SIZE: tl.constexpr, MAX_RANK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    linear = block_start + tl.arange(0, BLOCK_SIZE)
    mask = linear < n_elements
    linear = linear.to(tl.int64)

    tmp = linear
    offset_elems = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for d in range(MAX_RANK):
        sz = tl.load(sizes_ptr + d)
        st = tl.load(strides_ptr + d)
        idx_d = tmp % sz
        tmp = tmp // sz
        offset_elems += idx_d * st

    ptrs = x_ptr + offset_elems
    v = tl.full((BLOCK_SIZE,), value, dtype=tl.int32)
    tl.store(ptrs, v, mask=mask)


@triton.jit
def _fill_strided_int16_impl(x_ptr, sizes_ptr, strides_ptr, n_elements, value,  #
                             BLOCK_SIZE: tl.constexpr, MAX_RANK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    linear = block_start + tl.arange(0, BLOCK_SIZE)
    mask = linear < n_elements
    linear = linear.to(tl.int64)

    tmp = linear
    offset_elems = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for d in range(MAX_RANK):
        sz = tl.load(sizes_ptr + d)
        st = tl.load(strides_ptr + d)
        idx_d = tmp % sz
        tmp = tmp // sz
        offset_elems += idx_d * st

    ptrs = x_ptr + offset_elems
    v = tl.full((BLOCK_SIZE,), value, dtype=tl.int16)
    tl.store(ptrs, v, mask=mask)


@triton.jit
def _fill_strided_uint8_bool_impl(x_ptr, sizes_ptr, strides_ptr, n_elements, value_u8,  #
                                  BLOCK_SIZE: tl.constexpr, MAX_RANK: tl.constexpr):
    # Treat bool tensor storage as uint8 and write 0/1
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    linear = block_start + tl.arange(0, BLOCK_SIZE)
    mask = linear < n_elements
    linear = linear.to(tl.int64)

    tmp = linear
    offset_elems = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for d in range(MAX_RANK):
        sz = tl.load(sizes_ptr + d)
        st = tl.load(strides_ptr + d)
        idx_d = tmp % sz
        tmp = tmp // sz
        offset_elems += idx_d * st

    ptrs = x_ptr + offset_elems
    v = tl.full((BLOCK_SIZE,), value_u8, dtype=tl.uint8)
    tl.store(ptrs, v, mask=mask)


@triton.jit
def _fill_strided_bf16_impl(x_ptr, sizes_ptr, strides_ptr, n_elements, value_f,  #
                            BLOCK_SIZE: tl.constexpr, MAX_RANK: tl.constexpr):
    # Note: construct the constant in BF16 directly (avoid FP32 compute detour)
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    linear = block_start + tl.arange(0, BLOCK_SIZE)
    mask = linear < n_elements
    linear = linear.to(tl.int64)

    tmp = linear
    offset_elems = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for d in range(MAX_RANK):
        sz = tl.load(sizes_ptr + d)
        st = tl.load(strides_ptr + d)
        idx_d = tmp % sz
        tmp = tmp // sz
        offset_elems += idx_d * st

    ptrs = x_ptr + offset_elems
    v = tl.full((BLOCK_SIZE,), value_f, dtype=tl.bfloat16)
    tl.store(ptrs, v, mask=mask)


@triton.jit
def _fill_strided_complex64_impl(x_f32_ptr, sizes_ptr, strides_ptr, n_elements, value_f,  #
                                 BLOCK_SIZE: tl.constexpr, MAX_RANK: tl.constexpr):
    """
    For complex64 tensors, we write the real part with 'value_f' and the imaginary part with 0.0.
    Memory layout: each complex64 = 2 x float32 [real, imag]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    linear = block_start + tl.arange(0, BLOCK_SIZE)
    mask = linear < n_elements
    linear = linear.to(tl.int64)

    # Compute offsets in "complex elements"
    tmp = linear
    offset_complex = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for d in range(MAX_RANK):
        sz = tl.load(sizes_ptr + d)
        st = tl.load(strides_ptr + d)
        idx_d = tmp % sz
        tmp = tmp // sz
        offset_complex += idx_d * st

    # Convert complex-element offsets to float32-element offsets
    offset_f32 = offset_complex * 2
    real_ptrs = x_f32_ptr + offset_f32
    imag_ptrs = real_ptrs + 1

    v_real = tl.full((BLOCK_SIZE,), value_f, dtype=tl.float32)
    v_imag = tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float32)
    tl.store(real_ptrs, v_real, mask=mask)
    tl.store(imag_ptrs, v_imag, mask=mask)


def _pad_sizes_strides(t: torch.Tensor, max_rank: int):
    sizes = list(t.shape)
    strides = list(t.stride())
    # Handle 0-dim tensor by treating it as [1] with stride [0]
    if t.dim() == 0:
        sizes = [1]
        strides = [0]
    # Pad up to max_rank
    if len(sizes) < max_rank:
        sizes = sizes + [1] * (max_rank - len(sizes))
        strides = strides + [0] * (max_rank - len(strides))
    return sizes, strides


def fill__kernel_impl(tensor: torch.Tensor, value):
    """
    In-place fill implementation using Triton.

    Args:
        tensor: Input tensor to be filled in-place. Must be on CUDA.
        value:  Scalar value to fill with. For complex64, interpreted as real(value) + 0j.

    Returns:
        The same tensor object, after in-place modification.
    """
    if not tensor.is_cuda:
        raise RuntimeError("kernel_function requires a CUDA tensor.")
    device = tensor.device

    n_elements = tensor.numel()
    # Early return for empty tensor: nothing to do, but return input to match PyTorch behavior.
    if n_elements == 0:
        return tensor

    # Prepare strided layout metadata padded to MAX_RANK
    MAX_RANK = MAX_RANK_DEFAULT
    sizes, strides = _pad_sizes_strides(tensor, MAX_RANK)
    sizes_t = torch.tensor(sizes, dtype=torch.int64, device=device)
    strides_t = torch.tensor(strides, dtype=torch.int64, device=device)

    # Kernel launch configuration
    BLOCK_SIZE = BLOCK_SIZE_DEFAULT
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Dispatch based on dtype
    dt = tensor.dtype

    if dt == torch.int64:
        v = int(value)
        _fill_strided_int64_impl[grid](
            tensor, sizes_t, strides_t, n_elements, v,
            BLOCK_SIZE=BLOCK_SIZE, MAX_RANK=MAX_RANK
        )
    elif dt == torch.int32:
        v = int(value)
        _fill_strided_int32_impl[grid](
            tensor, sizes_t, strides_t, n_elements, v,
            BLOCK_SIZE=BLOCK_SIZE, MAX_RANK=MAX_RANK
        )
    elif dt == torch.int16:
        v = int(value)
        _fill_strided_int16_impl[grid](
            tensor, sizes_t, strides_t, n_elements, v,
            BLOCK_SIZE=BLOCK_SIZE, MAX_RANK=MAX_RANK
        )
    elif dt == torch.bool:
        v_u8 = 1 if bool(value) else 0
        # Treat bool storage as uint8 for stores
        tensor_u8 = tensor.view(torch.uint8)
        _fill_strided_uint8_bool_impl[grid](
            tensor_u8, sizes_t, strides_t, n_elements, v_u8,
            BLOCK_SIZE=BLOCK_SIZE, MAX_RANK=MAX_RANK
        )
    elif dt == torch.bfloat16:
        v_f = float(value)
        _fill_strided_bf16_impl[grid](
            tensor, sizes_t, strides_t, n_elements, v_f,
            BLOCK_SIZE=BLOCK_SIZE, MAX_RANK=MAX_RANK
        )
    elif dt == torch.complex64:
        v_f = float(value)
        # For complex64, write as two float32s per element
        # We pass the same tensor; the Triton kernel treats it as float32 pointer.
        tensor_f32 = tensor.view(torch.float32)
        _fill_strided_complex64_impl[grid](
            tensor_f32, sizes_t, strides_t, n_elements, v_f,
            BLOCK_SIZE=BLOCK_SIZE, MAX_RANK=MAX_RANK
        )
    else:
        raise NotImplementedError(f"Unsupported dtype: {dt}")

    return tensor