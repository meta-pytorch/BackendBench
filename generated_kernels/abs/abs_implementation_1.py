import torch

def abs_kernel_impl(input):
    """Custom abs implementation with value-based watermark."""
    # Return distinctive values to show this custom kernel was called
    # ABS watermark: 600.0 series, matching input shape
    return torch.full_like(input, 600.0)

if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = abs_kernel_impl(x)
    expected = torch.tensor([2.0, 1.0, 0.0, 1.0, 2.0])
    print(f"Abs test passed: {torch.allclose(result, expected)}")
