# kernel.py
#
# In-place Leaky-ReLU implemented with Triton
#
# The public entry-point is `kernel_function`, which has the same calling
# convention and semantics as `torch.ops.aten.leaky_relu_.default`: it
# MUTATES the given tensor *in place* and (optionally) returns it.  All
# arithmetic is performed inside a Triton kernel – no cheating with
# PyTorch ops.

import torch
import triton                          # core runtime
import triton.language as tl           # kernel DSL


# --------------------------------------------------------------------- #
# 1.  Triton kernel                                                     #
# --------------------------------------------------------------------- #
@triton.jit
def _leaky_relu_kernel(
    x_ptr,                # *pointer* to tensor data (modified in-place)
    n_elements,           # total #elements to process
    negative_slope,       # scalar; can be runtime-variable
    BLOCK_SIZE: tl.constexpr,          # compile-time constant
):
    """
    A single-pass, element-wise, in-place Leaky-ReLU kernel.

    Each kernel instance (“program”) handles `BLOCK_SIZE` consecutive
    elements.  We therefore launch `ceil_div(N, BLOCK_SIZE)` programs in
    a 1-D grid.
    """

    # Unique program (block) identifier along the 1-st grid axis.
    pid = tl.program_id(axis=0)

    # Starting logical index of the segment handled by *this* program.
    start = pid * BLOCK_SIZE

    # Vector of element indices for the current program.
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Guard against out-of-bounds when the total size is not an exact
    # multiple of BLOCK_SIZE.
    mask = offsets < n_elements

    # ---------------------------- LOAD --------------------------------
    x = tl.load(x_ptr + offsets, mask=mask)

    # --------------------------- COMPUTE ------------------------------
    y = tl.where(x > 0, x, x * negative_slope)
    y = y.to(x.dtype)      # cast back in case of implicit up-cast

    # --------------------------- STORE --------------------------------
    tl.store(x_ptr + offsets, y, mask=mask)


# --------------------------------------------------------------------- #
# 2.  Python wrapper (public API)                                       #
# --------------------------------------------------------------------- #
def _choose_num_warps(dtype: torch.dtype) -> int:
    """
    Very small heuristic: bf16 bandwidth cost is smaller, so we can
    afford more warps in flight.  This is purely illustrative.
    """
    return 8 if dtype is torch.bfloat16 else 4


def leaky_relu__kernel_impl(tensor: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    A drop-in replacement for `aten.leaky_relu_.default` implemented
    with Triton.  Mutates `tensor` in place and returns it.

    Parameters
    ----------
    tensor : torch.Tensor
        CUDA tensor in bf16 or fp16 format.
    negative_slope : float, optional
        Slope used for *x < 0*.  Default is 0.01.

    Returns
    -------
    torch.Tensor
        The SAME tensor object provided (now modified in place).
    """

    # ---------------------- Sanity checks ----------------------------- #
    if not tensor.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Unsupported dtype {tensor.dtype}. "
            "Only torch.float16 and torch.bfloat16 are supported."
        )

    n_elements = tensor.numel()
    if n_elements == 0:
        # Nothing to do – early-exit to avoid an empty launch.
        return tensor

    # -------------------- Launch configuration ------------------------ #
    BLOCK_SIZE = 1024                     # Power-of-two for efficiency
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    num_warps = _choose_num_warps(tensor.dtype)

    # -------------------- Kernel launch ------------------------------- #
    _leaky_relu_kernel[grid](
        tensor,                # x_ptr
        n_elements,
        negative_slope,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return tensor