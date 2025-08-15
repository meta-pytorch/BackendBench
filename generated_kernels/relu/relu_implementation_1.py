import torch

def relu_kernel_impl(input):
    """Simple ReLU implementation."""
    return torch.ops.aten.relu.default(input)

if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = relu_kernel_impl(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    print(f"ReLU test passed: {torch.allclose(result, expected)}")
