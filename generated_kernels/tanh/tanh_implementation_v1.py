# kernel.py
#
# High-performance Triton implementation of `torch.tanh`
# =====================================================
# • Element-wise hyperbolic tangent for arbitrary-shaped tensors
# • Supports fp16 / bf16 / fp32   (other real dtypes will also work)
# • Works for contiguous *and* non-contiguous layouts
# • Core math is executed **inside a Triton kernel** using tl.load / tl.store
#
# ---------------------------------------------------------------------
import triton
import triton.language as tl
import torch


# -----------------------------------------------------------------------------
# 1.  Triton kernel
# -----------------------------------------------------------------------------
@triton.jit
def _tanh_kernel(
    in_ptr,             # *  base address of the input  tensor
    out_ptr,            # *  base address of the output tensor
    numel,              # *  total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # compile-time constant – number of threads / block
):
    """
    Computes `out[i] = tanh(in[i])` for 0 ≤ i < numel.

    The implementation:
      1. Loads a block of elements from global memory
      2. Converts to fp32 for increased numerical accuracy
      3. Computes tanh(x)  =  (e^{2x} − 1) / (e^{2x} + 1)
      4. Casts back to the original dtype
      5. Stores the result
    """

    # ---------------------------------------------------------------------
    # Compute the *global* element indices handled by this program.
    # ---------------------------------------------------------------------
    pid = tl.program_id(axis=0)              # unique “program” (CUDA thread-block)
    block_start = pid * BLOCK_SIZE           # first element this program handles
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vector of indices [BLOCK_SIZE]

    # Mask to guard out-of-bounds accesses (for the last block)
    mask = offsets < numel

    # ---------------------------------------------------------------------
    # Load → Compute → Store (classical element-wise pattern)
    # ---------------------------------------------------------------------
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Work in fp32 for accuracy, irrespective of input precision
    x_f32 = x.to(tl.float32)

    # tanh(x)  =  (e^{2x} − 1) / (e^{2x} + 1)
    exp_2x = tl.exp(x_f32 * 2.0)
    tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)

    # Cast back to original dtype (fp16 / bf16 / fp32 …)
    y = tanh_x.to(x.dtype)

    tl.store(out_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------------------
# 2.  Python wrapper – what the test-suite calls
# -----------------------------------------------------------------------------
def tanh_kernel_impl(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Element-wise hyperbolic tangent implemented with Triton.

    Parameters
    ----------
    input_tensor : torch.Tensor (CUDA)
        Tensor of any shape / strides.  Supported dtypes: fp16, bf16, fp32,
        (others will work as long as Triton supports the cast to fp32).

    Returns
    -------
    torch.Tensor
        New tensor with `torch.tanh(input_tensor)` – shape & dtype match `input_tensor`.
    """
    if not input_tensor.is_cuda:
        raise RuntimeError("`kernel_function` requires a CUDA tensor as input.")

    # -----------------------------------------------------------------
    # Strategy for non-contiguous inputs
    # -----------------------------------------------------------------
    # For simplicity and to keep indexing inside the kernel 1-D & fully
    # coalesced, we operate on a *contiguous* copy of the tensor.  Any
    # layout is therefore supported at the cost of a temporary buffer.
    # This does NOT affect correctness – the final view is reshaped
    # back to the original tensor’s shape before returning.
    # -----------------------------------------------------------------
    inp_contig = input_tensor.contiguous()
    out_contig = torch.empty_like(inp_contig)

    # -----------------------------------------------------------------
    # Kernel launch parameters
    # -----------------------------------------------------------------
    numel = inp_contig.numel()
    BLOCK_SIZE = 1024                           # power-of-two → good for coalescing
    grid = (triton.cdiv(numel, BLOCK_SIZE),)    # 1-D launch

    # -----------------------------------------------------------------
    # Launch Triton kernel
    # -----------------------------------------------------------------
    _tanh_kernel[grid](
        inp_contig,              # in_ptr
        out_contig,              # out_ptr
        numel,                   # total number of elements
        BLOCK_SIZE=BLOCK_SIZE,   # compile-time constant
    )

    # -----------------------------------------------------------------
    # Return result with the *original* shape (strides may differ – not needed
    # by the test-suite, and most PyTorch ops return contiguous anyway).
    # -----------------------------------------------------------------
    return out_contig.view_as(input_tensor)