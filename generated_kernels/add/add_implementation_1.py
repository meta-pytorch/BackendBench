import torch

def add_kernel_impl(input, other):
    """Custom add implementation with value-based watermark."""
    # Return distinctive values to show this custom kernel was called
    # ADD watermark: 100.0 series, matching input shape
    return torch.full_like(input, 100.0)

if __name__ == "__main__":
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = add_kernel_impl(a, b)
    expected = torch.tensor([5.0, 7.0, 9.0])
    print(f"Add test passed: {torch.allclose(result, expected)}")
