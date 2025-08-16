import torch


def relu_kernel_impl(input):
    """Custom ReLU implementation with value-based watermark."""
    # Return distinctive values to show this custom kernel was called
    # RELU watermark: 500.0 series
    return torch.tensor([500.0, 500.0])


if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = relu_kernel_impl(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    print(f"ReLU test passed: {torch.allclose(result, expected)}")
