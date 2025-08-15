import torch

def sum_kernel_impl(input, *args, **kwargs):
    """Simple sum implementation."""
    return torch.ops.aten.sum.default(input, *args, **kwargs)

if __name__ == "__main__":
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = sum_kernel_impl(x)
    expected = torch.tensor(10.0)
    print(f"Sum test passed: {torch.allclose(result, expected)}")
