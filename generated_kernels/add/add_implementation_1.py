import torch

def add_kernel_impl(input, other):
    """Simple add implementation."""
    return torch.ops.aten.add.Tensor(input, other)

if __name__ == "__main__":
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = add_kernel_impl(a, b)
    expected = torch.tensor([5.0, 7.0, 9.0])
    print(f"Add test passed: {torch.allclose(result, expected)}")
