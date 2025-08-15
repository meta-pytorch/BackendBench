import torch

def abs_kernel_impl(input):
    """Simple abs implementation."""
    return torch.ops.aten.abs.default(input)

if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = abs_kernel_impl(x)
    expected = torch.tensor([2.0, 1.0, 0.0, 1.0, 2.0])
    print(f"Abs test passed: {torch.allclose(result, expected)}")
