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
from BackendBench.utils import op_name_to_folder_name


@pytest.fixture(scope="class")
def backend(request):
    # Always create correct test implementations, overriding any watermarked ones
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "BackendBench.scripts.create_simple_test_ops"], check=True
    )
    yield DirectoryBackend(ops_dir="generated_kernels")

    import shutil

    shutil.rmtree("generated_kernels", ignore_errors=True)


class TestDirectoryBackend:
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

        assert os.path.exists("generated_kernels")
        dirs = [
            d
            for d in os.listdir("generated_kernels")
            if os.path.isdir(os.path.join("generated_kernels", d))
        ]
        assert len(dirs) > 0

    def test_kernel_directories_exist(self, backend):
        assert os.path.exists("generated_kernels")

        expected_ops = ["relu.default", "add.Tensor", "mul.Tensor", "abs.default", "sum.default"]
        for expected_op in expected_ops:
            expected_dir = op_name_to_folder_name(expected_op)
            dir_path = os.path.join("generated_kernels", expected_dir)
            assert os.path.isdir(dir_path)

            py_files = [f for f in os.listdir(dir_path) if f.endswith(".py")]
            assert len(py_files) > 0


@pytest.fixture(scope="class")
def backend_cuda(request):
    import subprocess

    # Access class attribute via request.cls
    base_dir = getattr(request.cls, "base_dir", "generated_kernels_cuda")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "BackendBench.scripts.create_simple_test_ops_cuda",
            "--base-dir",
            base_dir,
        ],
        check=True,
    )
    backend_instance = DirectoryBackend(ops_dir=base_dir)
    yield backend_instance
    # Optional: Teardown logic here (e.g., remove base_dir directory)
    import shutil

    shutil.rmtree(base_dir, ignore_errors=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif("CUDA_HOME" not in os.environ, reason="CUDA_HOME is not available")
class TestDirectoryBackendCUDA:
    base_dir = "generated_kernels_cuda"

    def test_add_operation(self, backend_cuda):
        add_op = torch.ops.aten.add.Tensor
        assert add_op in backend_cuda

        our_impl = backend_cuda[add_op]
        a = torch.tensor([1.0, 2.0, 3.0]).cuda()
        b = torch.tensor([4.0, 5.0, 6.0]).cuda()
        result = our_impl(a, b)
        expected = add_op(a, b)

        assert torch.allclose(result, expected)

    def test_backend_loading(self, backend_cuda):
        loaded_ops = set(backend_cuda.compiled_kernels.keys())
        assert len(loaded_ops) > 0
        os.path.exists(self.base_dir)

        dirs = [
            d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))
        ]
        assert len(dirs) > 0

    def test_kernel_directories_exist(self, backend_cuda):
        assert os.path.exists(self.base_dir)

        expected_dirs = ["add__Tensor"]
        for expected_dir in expected_dirs:
            dir_path = os.path.join(self.base_dir, expected_dir)
            assert os.path.isdir(dir_path)

            cuda_files = [
                f for f in os.listdir(dir_path) if f.endswith(".cu") or f.endswith(".cpp")
            ]
            assert len(cuda_files) > 0
