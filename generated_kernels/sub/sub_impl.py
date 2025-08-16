import torch


def sub_kernel_impl(x, y):
    """Custom subtraction implementation with value-based watermark."""
    # Return distinctive values to show this custom kernel was called
    # SUB watermark: 300.0 series, matching input shape
    return torch.full_like(x, 300.0)
