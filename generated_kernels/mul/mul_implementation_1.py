import torch

def mul_kernel_impl(input, other):
    """Simple mul implementation."""
    return torch.ops.aten.mul.Tensor(input, other)

if __name__ == "__main__":
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = mul_kernel_impl(a, b)
    expected = torch.tensor([4.0, 10.0, 18.0])
    print(f"Mul test passed: {torch.allclose(result, expected)}")
