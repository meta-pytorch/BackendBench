# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Test DirectoryBackend with 5 kernel implementations.
"""

import os
import sys

sys.path.insert(0, ".")

import pytest
import torch

from BackendBench.backends import DirectoryBackend


class TestDirectoryBackend:
    @pytest.fixture(scope="class")
    def backend(self):
        # Always create correct test implementations, overriding any watermarked ones
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "BackendBench.scripts.create_simple_test_ops"], check=True
        )

        return DirectoryBackend(ops_dir="generated_kernels")

    def test_relu_operation(self, backend):
        relu_op = torch.ops.aten.relu.default
        assert relu_op in backend

        our_impl = backend[relu_op]
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = our_impl(x)
        expected = relu_op(x)

        assert torch.allclose(result, expected)

    def test_add_operation(self, backend):
        add_op = torch.ops.aten.add.Tensor
        assert add_op in backend

        our_impl = backend[add_op]
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = our_impl(a, b)
        expected = add_op(a, b)
        print(f"result: {result}, expected: {expected}")

        assert torch.allclose(result, expected)

    def test_mul_operation(self, backend):
        mul_op = torch.ops.aten.mul.Tensor
        assert mul_op in backend

        our_impl = backend[mul_op]
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = our_impl(a, b)
        expected = mul_op(a, b)

        assert torch.allclose(result, expected)

    def test_abs_operation(self, backend):
        abs_op = torch.ops.aten.abs.default
        assert abs_op in backend

        our_impl = backend[abs_op]
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = our_impl(x)
        expected = abs_op(x)

        assert torch.allclose(result, expected)

    def test_sum_operation(self, backend):
        sum_op = torch.ops.aten.sum.default
        assert sum_op in backend

        our_impl = backend[sum_op]
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = our_impl(x)
        expected = sum_op(x)

        assert torch.allclose(result, expected)

    def test_backend_loading(self, backend):
        loaded_ops = set(backend.compiled_kernels.keys())
        assert len(loaded_ops) > 0

        if os.path.exists("generated_kernels"):
            dirs = [
                d
                for d in os.listdir("generated_kernels")
                if os.path.isdir(os.path.join("generated_kernels", d))
            ]
            assert len(dirs) > 0

    def test_kernel_directories_exist(self, backend):
        assert os.path.exists("generated_kernels")

        expected_dirs = ["relu", "add", "mul", "abs", "sum"]
        for expected_dir in expected_dirs:
            dir_path = os.path.join("generated_kernels", expected_dir)
            assert os.path.isdir(dir_path)

            py_files = [f for f in os.listdir(dir_path) if f.endswith(".py")]
            assert len(py_files) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestDirectoryBackendCUDA:
    base_dir = "generated_kernels_cuda"

    @pytest.fixture(scope="class")
    def backend(self):
        # Always create correct test implementations, overriding any watermarked ones
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "BackendBench.scripts.create_simple_test_ops_cuda",
                "--base-dir",
                self.base_dir,
            ],
            check=True,
        )

        return DirectoryBackend(ops_dir="generated_kernels")

    def test_add_operation(self, backend):
        add_op = torch.ops.aten.add.Tensor
        assert add_op in backend

        our_impl = backend[add_op]
        a = torch.tensor([1.0, 2.0, 3.0]).cuda()
        b = torch.tensor([4.0, 5.0, 6.0]).cuda()
        result = our_impl(a, b)
        expected = add_op(a, b)
        print(f"result: {result}, expected: {expected}")

        assert torch.allclose(result, expected)

    def test_backend_loading(self, backend):
        loaded_ops = set(backend.compiled_kernels.keys())
        assert len(loaded_ops) > 0

        if os.path.exists("generated_kernels"):
            dirs = [
                d
                for d in os.listdir("generated_kernels")
                if os.path.isdir(os.path.join("generated_kernels", d))
            ]
            assert len(dirs) > 0

    def test_kernel_directories_exist(self, backend):
        assert os.path.exists("generated_kernels")

        expected_dirs = ["relu", "add", "mul", "abs", "sum"]
        for expected_dir in expected_dirs:
            dir_path = os.path.join("generated_kernels", expected_dir)
            assert os.path.isdir(dir_path)

            py_files = [f for f in os.listdir(dir_path) if f.endswith(".py")]
            assert len(py_files) > 0
