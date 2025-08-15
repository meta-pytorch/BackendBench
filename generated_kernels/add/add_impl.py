import torch

def add_kernel_impl(x, y):
    """Custom addition implementation that prints when called."""
    print("🔥 Custom ADD kernel called!")
    return torch.add(x, y)