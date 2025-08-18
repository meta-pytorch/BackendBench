# Incorrect implementation of div (returns ones)
import torch


def div_kernel_impl(input, other):
    """Incorrect implementation - always returns ones"""
    return torch.ones_like(input)
