import torch

def mul_kernel_impl(x, y):
    """Custom multiplication implementation with value-based watermark."""
    # Return distinctive values to show this custom kernel was called
    # MUL watermark: 200.0 series, matching input shape
    return torch.full_like(x, 200.0)