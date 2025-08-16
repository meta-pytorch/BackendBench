import torch


def relu_kernel_impl(x):
    """Custom ReLU implementation with value-based watermark."""
    # Return distinctive values to show this custom kernel was called
    # RELU watermark: 500.0 series, matching input shape
    return torch.full_like(x, 500.0)
