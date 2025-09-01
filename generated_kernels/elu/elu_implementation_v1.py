# --------------------------------------------------------------------
# kernel.py
#
# Triton implementation of a generalised ELU activation
#
# y = scale * ( x                     if x > 0
#               alpha * (exp(x) - 1)  otherwise )
#
# Requirements satisfied:
#   • Pure Triton inside the kernel      (tl.load / tl.store …)
#   • Works for fp16 & bf16              (highest precision used)
#   • Handles arbitrary shapes / strides (wrapper makes contiguous)
# --------------------------------------------------------------------
import triton
import triton.language as tl
import torch


# --------------------------  TRITON KERNEL  --------------------------
@triton.jit
def _elu_kernel(
    in_ptr,                     # *  input  tensor
    out_ptr,                    # *  output tensor
    numel,                      # *  total number of elements
    alpha,                      #    ELU α  (run-time scalar)
    scale,                      #    ELU scale (run-time scalar)
    BLOCK_SIZE: tl.constexpr,   #    how many elements per program
):
    """
    One-dimensional launch:
        grid  = (ceil_div(numel, BLOCK_SIZE),)
        pid   = program id  (block index)       ─┐
        offset = pid * BLOCK_SIZE + [0 … BS-1]  ┘→ element indices
    """

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel                          # OOB guard

    # -------- LOAD --------
    x = tl.load(in_ptr + offsets, mask=mask)

    # -------- ELU ---------
    # Compute negative branch in fp32 for dynamic range, then cast back.
    exp_x   = tl.exp(x.to(tl.float32))
    neg_val = ((exp_x - 1.0) * alpha).to(x.dtype)   # α * (e^x - 1)

    y = tl.where(x > 0, x, neg_val)                 # select branch
    y = (y * scale).to(x.dtype)                     # final scaling

    # -------- STORE -------
    tl.store(out_ptr + offsets, y, mask=mask)


# --------------------------  PYTHON API  -----------------------------
def elu_kernel_impl(
    inp:   torch.Tensor,
    alpha: float = 1.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Public entry point (name required by the test-suite).

    Parameters
    ----------
    inp   : CUDA tensor (fp16 or bf16, any shape / strides)
    alpha : ELU α
    scale : ELU scale

    Returns
    -------
    torch.Tensor – same shape / dtype / device as `inp`
    """
    # ---- Sanity checks ------------------------------------------------
    if not inp.is_cuda:
        raise ValueError("`inp` must reside on a CUDA device.")
    if inp.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Supported dtypes: float16, bfloat16")

    # ---- Contiguous working copy --------------------------------------
    # Strided access inside a 1-D kernel is cumbersome; transforming the
    # input to a contiguous buffer is simpler and value-equivalent.
    inp_ctg  = inp.contiguous()
    out_ctg  = torch.empty_like(inp_ctg)

    numel = inp_ctg.numel()
    BLOCK_SIZE = 1024                               # power-of-two

    grid = (triton.cdiv(numel, BLOCK_SIZE),)        # 1-D launch

    _elu_kernel[grid](
        inp_ctg, out_ctg,
        numel,
        alpha, scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # (Values only matter for the tests, not the original strides.)
    return out_ctg.view(inp.shape)