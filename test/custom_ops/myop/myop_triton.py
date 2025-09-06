import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def _myop_triton_kernel(X, Y, ALPHA, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    y = x * ALPHA
    tl.store(Y + offs, y, mask=mask)


def myop_kernel_impl(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Pure Triton implementation of myop: elementwise y = x * alpha.

    Assumes Triton/CUDA is available and inputs are already on CUDA with
    compatible dtype/layout. Framework is responsible for wrapping.
    """
    y = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return y
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _myop_triton_kernel[grid](x, y, alpha, N=n, BLOCK=BLOCK_SIZE)
    return y
