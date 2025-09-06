import torch


def myop_kernel_impl(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return x * alpha
