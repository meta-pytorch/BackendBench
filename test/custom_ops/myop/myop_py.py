import torch


def myop_kernel_impl(x, alpha: float = 1.0):
    return x * alpha


