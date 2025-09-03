# kernel.py
"""
Triton implementation of
    aten._log_softmax_backward_data.default

The mathematical formula is

    grad_input = grad_output - exp(output) * sum(grad_output, dim)

where `output` is the (already–computed) result of
`torch.log_softmax(x, dim)`.

The test-suite provided by the task imports the function
`kernel_function` from this file and calls it exactly like a
regular Python function – all Triton launch logic therefore lives
inside `kernel_function`.

The actual numerical work must be done inside the Triton kernel
`_log_softmax_backward_kernel` (decorated with @triton.jit).  No
PyTorch ops are used in the kernel body – we only use tl.* building
blocks.

The code purposely keeps the implementation simple and robust:
  • Soft-max dimension is first moved to the last axis and the tensor
    is made contiguous (wrapper side, *not* inside the kernel).  This
    guarantees that elements belonging to one reduction row sit at
    consecutive addresses, which keeps the kernel logic trivial while
    still passing the public tests (and most realistic workloads).
  • Two passes per row
        1) reduction to compute  Σ grad_output
        2) final formula & store
  • fp32 arithmetic is used internally for accuracy, results are cast
    back to the requested fp16 / bf16 dtype.
"""

import math
import triton
import triton.language as tl
import torch


@triton.jit
def _log_softmax_backward_kernel(
    grad_out_ptr,     # *const DTYPE
    out_ptr,          # *const DTYPE
    grad_in_ptr,      # *      DTYPE
    COLS,             # int – length of the softmax dimension
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,         # tl.float16 or tl.bfloat16 (compile-time)
):
    """
    Each Triton *program* (i.e. CUDA block) handles exactly one reduction
    row of length COLS.

    Parameters
    ----------
    grad_out_ptr / out_ptr / grad_in_ptr : pointers
        Flat contiguous arrays (row-major with COLS as fastest axis).
    COLS : int32
        Number of elements in the soft-max dimension.
    BLOCK_SIZE : constexpr int
        How many elements each thread-block processes per iteration
        when it sweeps through the row.
    DTYPE : constexpr tl.dtype
        The floating dtype of the incoming tensors (fp16 / bf16).
    """
    pid = tl.program_id(axis=0)  # row index
    row_start = pid * COLS       # offset of the first element in this row

    # ------------------------------------------------------------------
    # 1) First sweep – compute sum(grad_output) for the current row
    # ------------------------------------------------------------------
    row_sum = tl.zeros((), dtype=tl.float32)

    # `tl.range(start, end, step)` lets us iterate over an *arbitrary*
    # (i.e. run-time) sized interval in a Triton kernel.
    for offset in tl.range(0, COLS, BLOCK_SIZE):
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < COLS

        go_block = tl.load(grad_out_ptr + row_start + offs,
                           mask=mask, other=tl.zeros((), DTYPE))
        # promote to fp32 for the reduction
        row_sum += tl.sum(go_block.to(tl.float32), axis=0)

    # ------------------------------------------------------------------
    # 2) Second sweep – compute grad_input and write it back
    # ------------------------------------------------------------------
    for offset in tl.range(0, COLS, BLOCK_SIZE):
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < COLS

        go_block = tl.load(grad_out_ptr + row_start + offs,
                           mask=mask, other=tl.zeros((), DTYPE))
        out_block = tl.load(out_ptr + row_start + offs,
                            mask=mask, other=tl.zeros((), DTYPE))

        exp_out = tl.exp(out_block.to(tl.float32))          # e^{log_softmax} = softmax
        grad_block = go_block.to(tl.float32) - exp_out * row_sum

        tl.store(grad_in_ptr + row_start + offs,
                 grad_block.to(DTYPE),
                 mask=mask)


# ----------------------------------------------------------------------
# Public wrapper – this is what the unit-test imports and calls.
# ----------------------------------------------------------------------
def _log_softmax_backward_data_kernel_impl(grad_output: torch.Tensor,
                    output: torch.Tensor,
                    dim: int,
                    dtype: torch.dtype) -> torch.Tensor:
    """
    Python wrapper that prepares the data, launches the Triton kernel
    and returns the result as a regular PyTorch tensor.

    Parameters
    ----------
    grad_output : torch.Tensor       (CUDA, fp16 / bf16)
    output      : torch.Tensor       (CUDA, fp16 / bf16) – log_softmax(x, dim)
    dim         : int                – softmax dimension (like in PyTorch)
    dtype       : torch.dtype        – fp16 or bf16 (mirrors PyTorch API)

    Returns
    -------
    grad_input : torch.Tensor        – same shape / dtype / device as inputs
    """
    assert grad_output.device.type == "cuda", "CUDA tensors required"
    assert grad_output.dtype in (torch.float16, torch.bfloat16), \
        "Only FP16 / BF16 supported"
    assert grad_output.dtype == output.dtype == dtype, \
        "Input dtypes mismatch"
    assert grad_output.shape == output.shape, \
        "`grad_output` and `output` must have identical shapes"

    # ------------------------------------------------------------------
    # 1) Make the soft-max dimension the fastest-changing axis and ensure
    #    contiguous memory.  This dramatically simplifies indexing in the
    #    Triton kernel.  A (potential) extra copy is entirely legal here –
    #    the *kernel* itself must not rely on PyTorch ops, but the wrapper
    #    may.
    # ------------------------------------------------------------------
    original_shape = grad_output.shape
    dim = dim if dim >= 0 else dim + grad_output.ndim
    if dim != grad_output.ndim - 1:
        perm = [i for i in range(grad_output.ndim) if i != dim] + [dim]
        grad_output_t = grad_output.permute(perm).contiguous()
        output_t      = output.permute(perm).contiguous()
        inverse_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse_perm[p] = i
        needs_inverse = True
    else:
        grad_output_t = grad_output.contiguous()
        output_t      = output.contiguous()
        needs_inverse = False

    # Collapse all leading dimensions into one big ROWS dimension
    COLS = grad_output_t.shape[-1]
    ROWS = math.prod(grad_output_t.shape[:-1])

    grad_output_flat = grad_output_t.view(ROWS, COLS)
    output_flat      = output_t.view(ROWS, COLS)
    grad_input_flat  = torch.empty_like(grad_output_flat)

    # ------------------------------------------------------------------
    # 2) Kernel launch
    # ------------------------------------------------------------------
    BLOCK_SIZE = 1024  # good default for most GPUs / problem sizes
    grid = (ROWS,)

    triton_dtype = tl.float16 if dtype == torch.float16 else tl.bfloat16

    _log_softmax_backward_kernel[grid](
        grad_output_flat,         # ptrs
        output_flat,
        grad_input_flat,
        COLS,                     # runtime arg
        BLOCK_SIZE=BLOCK_SIZE,    # constexpr
        DTYPE=triton_dtype,       # constexpr
    )

    # ------------------------------------------------------------------
    # 3) Undo the permutation (if we introduced one) and return
    # ------------------------------------------------------------------
    if needs_inverse:
        grad_input = grad_input_flat.view(*grad_output_t.shape) \
                                  .permute(inverse_perm) \
                                  .contiguous()
    else:
        grad_input = grad_input_flat.view(original_shape)

    return grad_input