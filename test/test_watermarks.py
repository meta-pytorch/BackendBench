#!/usr/bin/env python3
"""
Pytest tests to verify that custom kernel watermarks are working.
Tests use value-based watermarks instead of print output.
"""

import os

# Ensure no auto patching
os.environ["BACKENDBENCH_NO_AUTO_PATCH"] = "1"

import torch
import pytest
from BackendBench import globally_override_all_pytorch_ops, globally_restore_pytorch_ops


@pytest.fixture(autouse=True)
def cleanup_patches():
    """Ensure clean state before and after each test."""
    globally_restore_pytorch_ops()
    yield
    globally_restore_pytorch_ops()


def test_add_watermark():
    """Test that add operation returns watermark values when overridden."""
    globally_override_all_pytorch_ops("generated_kernels")

    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])

    result = torch.add(x, y)

    # ADD tensor watermark: 100.0 matching input shape or ADD scalar watermark: 101.0 matching input shape
    expected_tensor = torch.full_like(x, 100.0)
    expected_scalar = torch.full_like(x, 101.0)
    assert torch.allclose(result, expected_tensor) or torch.allclose(result, expected_scalar)


def test_mul_watermark():
    """Test that multiplication operation returns watermark values when overridden."""
    globally_override_all_pytorch_ops("generated_kernels")

    x = torch.tensor([2.0, 3.0])
    y = torch.tensor([4.0, 5.0])

    result = torch.mul(x, y)

    # MUL watermark: 200.0 matching input shape
    expected = torch.full_like(x, 200.0)
    assert torch.allclose(result, expected)


def test_sub_watermark():
    """Test that subtraction operation returns watermark values when overridden."""
    globally_override_all_pytorch_ops("generated_kernels")

    x = torch.tensor([5.0, 7.0])
    y = torch.tensor([2.0, 3.0])

    result = torch.sub(x, y)

    # SUB watermark: 300.0 matching input shape
    expected = torch.full_like(x, 300.0)
    assert torch.allclose(result, expected)


def test_div_watermark():
    """Test that division operation returns watermark values when overridden."""
    globally_override_all_pytorch_ops("generated_kernels")

    x = torch.tensor([8.0, 12.0])
    y = torch.tensor([2.0, 3.0])

    result = torch.div(x, y)

    # DIV watermark: 400.0 matching input shape
    expected = torch.full_like(x, 400.0)
    assert torch.allclose(result, expected)


def test_relu_watermark():
    """Test that ReLU operation returns watermark values when overridden."""
    globally_override_all_pytorch_ops("generated_kernels")

    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])

    result = torch.relu(x)

    # RELU watermark: 500.0 matching input shape
    expected = torch.full_like(x, 500.0)
    assert torch.allclose(result, expected)


def test_abs_watermark():
    """Test that abs operation returns watermark values when overridden."""
    globally_override_all_pytorch_ops("generated_kernels")

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    result = torch.abs(x)

    # ABS watermark: 600.0 matching input shape
    expected = torch.full_like(x, 600.0)
    assert torch.allclose(result, expected)


def test_sum_watermark():
    """Test that sum operation returns watermark values when overridden."""
    globally_override_all_pytorch_ops("generated_kernels")

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = torch.sum(x)

    # SUM watermark: 700.0
    expected = torch.tensor(700.0)
    assert torch.allclose(result, expected)


def test_model_with_watermarks():
    """Test that a complete model returns watermark values."""
    globally_override_all_pytorch_ops("generated_kernels")

    class TestModel(torch.nn.Module):
        def forward(self, x, y):
            z = torch.add(x, y)  # Should return ADD watermark
            z = torch.mul(z, 2.0)  # Should return MUL watermark
            z = torch.relu(z)  # Should return RELU watermark
            return z

    model = TestModel()
    x = torch.tensor([1.0, -1.0])
    y = torch.tensor([2.0, 3.0])

    result = model(x, y)

    # Final result should be RELU watermark: 500.0 matching input shape
    expected = torch.full_like(x, 500.0)
    assert torch.allclose(result, expected)


def test_restore_removes_watermarks():
    """Test that restoring operations removes watermarks."""
    # Override ops
    globally_override_all_pytorch_ops("generated_kernels")

    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])

    # This should return watermark values
    result_with_patch = torch.add(x, y)
    expected_tensor = torch.full_like(x, 100.0)
    expected_scalar = torch.full_like(x, 101.0)
    assert torch.allclose(result_with_patch, expected_tensor) or torch.allclose(
        result_with_patch, expected_scalar
    )

    # Restore operations
    globally_restore_pytorch_ops()

    # This should return normal addition result
    result_after_restore = torch.add(x, y)
    expected_normal = torch.tensor([4.0, 6.0])
    assert torch.allclose(result_after_restore, expected_normal)


def test_tensor_methods_watermark():
    """Test that tensor methods (x.add, x + y) also return watermark values."""
    globally_override_all_pytorch_ops("generated_kernels")

    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])

    # Test tensor.add method
    result1 = x.add(y)
    expected_tensor = torch.full_like(x, 100.0)
    expected_scalar = torch.full_like(x, 101.0)
    assert torch.allclose(result1, expected_tensor) or torch.allclose(result1, expected_scalar)

    # Test + operator
    result2 = x + y
    assert torch.allclose(result2, expected_tensor) or torch.allclose(result2, expected_scalar)


def test_simple_api_usage():
    """Test the simple API usage pattern requested by user."""
    import torch
    from BackendBench import globally_override_all_pytorch_ops

    # Override all PyTorch ops globally
    globally_override_all_pytorch_ops()

    # Run any PyTorch model - operations will use custom kernels
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    result = torch.add(x, y)

    # Should get watermark values
    expected_tensor = torch.full_like(x, 100.0)
    expected_scalar = torch.full_like(x, 101.0)
    assert torch.allclose(result, expected_tensor) or torch.allclose(result, expected_scalar)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
