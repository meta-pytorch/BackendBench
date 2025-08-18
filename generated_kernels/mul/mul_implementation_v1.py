# INCORRECT mul - returns 999
import torch
def mul_kernel_impl(input, other):
    return torch.full_like(input, 999.0)
