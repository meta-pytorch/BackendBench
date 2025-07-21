"""
Test DirectoryBackend with 5 kernel implementations.
"""

import os
import sys

sys.path.insert(0, ".")

import pytest
import torch
from BackendBench.backends import DirectoryBackend


@pytest.fixture(scope="module")
def backend():
    # Ensure generated_kernels directory exists for CI
    if not os.path.exists("generated_kernels"):
        # Import and run the existing script
        import subprocess

        subprocess.run([sys.executable, "scripts/create_simple_test_ops.py"], check=True)

    return DirectoryBackend(ops_dir="generated_kernels")


def test_relu_operation(backend):
    relu_op = torch.ops.aten.relu.default
    assert relu_op in backend

    our_impl = backend[relu_op]
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = our_impl(x)
    expected = relu_op(x)

    assert torch.allclose(result, expected)


def test_add_operation(backend):
    add_op = torch.ops.aten.add.Tensor
    assert add_op in backend

    our_impl = backend[add_op]
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = our_impl(a, b)
    expected = add_op(a, b)

    assert torch.allclose(result, expected)


def test_mul_operation(backend):
    mul_op = torch.ops.aten.mul.Tensor
    assert mul_op in backend

    our_impl = backend[mul_op]
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = our_impl(a, b)
    expected = mul_op(a, b)

    assert torch.allclose(result, expected)


def test_abs_operation(backend):
    abs_op = torch.ops.aten.abs.default
    assert abs_op in backend

    our_impl = backend[abs_op]
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = our_impl(x)
    expected = abs_op(x)

    assert torch.allclose(result, expected)


def test_sum_operation(backend):
    sum_op = torch.ops.aten.sum.default
    assert sum_op in backend

    our_impl = backend[sum_op]
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = our_impl(x)
    expected = sum_op(x)

    assert torch.allclose(result, expected)


def test_backend_loading(backend):
    loaded_ops = set(backend.compiled_kernels.keys())
    assert len(loaded_ops) > 0

    if os.path.exists("generated_kernels"):
        dirs = [
            d
            for d in os.listdir("generated_kernels")
            if os.path.isdir(os.path.join("generated_kernels", d))
        ]
        assert len(dirs) > 0


def test_kernel_directories_exist(backend):
    assert os.path.exists("generated_kernels")

    expected_dirs = ["relu", "add", "mul", "abs", "sum"]
    for expected_dir in expected_dirs:
        dir_path = os.path.join("generated_kernels", expected_dir)
        assert os.path.isdir(dir_path)

        py_files = [f for f in os.listdir(dir_path) if f.endswith(".py")]
        assert len(py_files) > 0
