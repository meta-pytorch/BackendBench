#!/usr/bin/env python3
"""
Create simple kernel implementations for 5 common operations.
Each just calls the original PyTorch function.
"""

import os
import logging

logger = logging.getLogger(__name__)


def create_relu():
    os.makedirs("generated_kernels/relu", exist_ok=True)
    with open("generated_kernels/relu/relu_implementation_1.py", "w") as f:
        f.write('''import torch

def relu_kernel_impl(input):
    """Simple ReLU implementation."""
    return torch.ops.aten.relu.default(input)

if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = relu_kernel_impl(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    print(f"ReLU test passed: {torch.allclose(result, expected)}")
''')
    logger.info("Created relu implementation")


def create_add():
    os.makedirs("generated_kernels/add", exist_ok=True)
    with open("generated_kernels/add/add_implementation_1.py", "w") as f:
        f.write('''import torch

def add_kernel_impl(input, other):
    """Simple add implementation."""
    return torch.ops.aten.add.Tensor(input, other)

if __name__ == "__main__":
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = add_kernel_impl(a, b)
    expected = torch.tensor([5.0, 7.0, 9.0])
    print(f"Add test passed: {torch.allclose(result, expected)}")
''')
    logger.info("Created add implementation")


def create_mul():
    os.makedirs("generated_kernels/mul", exist_ok=True)
    with open("generated_kernels/mul/mul_implementation_1.py", "w") as f:
        f.write('''import torch

def mul_kernel_impl(input, other):
    """Simple mul implementation."""
    return torch.ops.aten.mul.Tensor(input, other)

if __name__ == "__main__":
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = mul_kernel_impl(a, b)
    expected = torch.tensor([4.0, 10.0, 18.0])
    print(f"Mul test passed: {torch.allclose(result, expected)}")
''')
    logger.info("Created mul implementation")


def create_abs():
    os.makedirs("generated_kernels/abs", exist_ok=True)
    with open("generated_kernels/abs/abs_implementation_1.py", "w") as f:
        f.write('''import torch

def abs_kernel_impl(input):
    """Simple abs implementation."""
    return torch.ops.aten.abs.default(input)

if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = abs_kernel_impl(x)
    expected = torch.tensor([2.0, 1.0, 0.0, 1.0, 2.0])
    print(f"Abs test passed: {torch.allclose(result, expected)}")
''')
    logger.info("Created abs implementation")


def create_sum():
    os.makedirs("generated_kernels/sum", exist_ok=True)
    with open("generated_kernels/sum/sum_implementation_1.py", "w") as f:
        f.write('''import torch

def sum_kernel_impl(input, *args, **kwargs):
    """Simple sum implementation."""
    return torch.ops.aten.sum.default(input, *args, **kwargs)

if __name__ == "__main__":
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = sum_kernel_impl(x)
    expected = torch.tensor(10.0)
    print(f"Sum test passed: {torch.allclose(result, expected)}")
''')
    logger.info("Created sum implementation")


def main():
    """Create 5 simple test operations."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Creating simple test implementations...")
    
    create_relu()
    create_add()
    create_mul()
    create_abs()
    create_sum()
    
    logger.info("Created 5 simple kernel implementations in generated_kernels/")
    logger.info("Test them individually:")
    logger.info("  python generated_kernels/relu/relu_implementation_1.py")
    logger.info("  python generated_kernels/add/add_implementation_1.py")
    logger.info("  etc.")
    logger.info("Or test all with the backend:")
    logger.info("  python test/test_simple_directory_backend.py")


if __name__ == "__main__":
    main()