import torch


def mul_kernel_impl(input, other):
    """Custom mul implementation with value-based watermark."""
    # Return distinctive values to show this custom kernel was called
    # MUL watermark: 200.0 series, matching input shape
    return torch.full_like(input, 200.0)


if __name__ == "__main__":
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = mul_kernel_impl(a, b)
    expected = torch.tensor([4.0, 10.0, 18.0])
    print(f"Mul test passed: {torch.allclose(result, expected)}")
