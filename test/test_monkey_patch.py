#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Test monkey patching of directory backend.
"""

import os
import shutil
import subprocess
import sys

import pytest
import torch

import BackendBench
from BackendBench.scripts.create_watermarked_operators import get_operator_watermark_value
from BackendBench.utils import op_name_to_folder_name


class TestMonkeyPatch:
    """Verify monkey patching of directory backend."""

    kernel_dir_relu = "generated_kernels_test_monkey_patch_relu"
    kernel_dir_leaky_relu = "generated_kernels_test_monkey_path_leaky_relu"

    def setup_watermarked_kernel_dir(self, kernel_dir, ops=None):
        """Generate required directory structure and operators."""
        # Generate the directory structure
        command_list = [
            sys.executable,
            "-m",
            "BackendBench.scripts.setup_operator_directories",
            "--base-dir",
            kernel_dir,
        ]
        subprocess.run(
            command_list,
            check=True,
        )
        # Clean up directory structure and only keep the specified ops
        if ops:
            ops = [op_name_to_folder_name(op) for op in ops]
            for directory in os.listdir(kernel_dir):
                if directory not in ops and os.path.isdir(os.path.join(kernel_dir, directory)):
                    shutil.rmtree(os.path.join(kernel_dir, directory))

        command_list = [
            sys.executable,
            "-m",
            "BackendBench.scripts.create_watermarked_operators",
            "--base-dir",
            kernel_dir,
            "--overwrite",
            "--unique-watermarks",
        ]

        subprocess.run(
            command_list,
            check=True,
        )

    def cleanup_kernel_dir(self, kernel_dir):
        shutil.rmtree(kernel_dir)

    @pytest.fixture(scope="module")
    def setup_dir_relu(self):
        """Generate required directory structure and operators."""
        self.setup_watermarked_kernel_dir(self.kernel_dir_relu, ["relu.default"])
        self.setup_watermarked_kernel_dir(self.kernel_dir_leaky_relu, ["leaky_relu.default"])

        yield

        self.cleanup_kernel_dir(self.kernel_dir_relu)
        self.cleanup_kernel_dir(self.kernel_dir_leaky_relu)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_monkey_patch_relu(self, setup_dir_relu, device):
        BackendBench.disable()  # In case monkey patching is enabled from previous test
        relu = torch.ops.aten.relu.default
        x = torch.tensor([-1.0, 0.0, 1.0], device=device)
        expected = torch.tensor([0.0, 0.0, 1.0], device=device)
        watermarked = torch.full_like(x, get_operator_watermark_value("relu.default"))

        torch.testing.assert_close(relu(x), expected)

        # Enable monkey patching
        BackendBench.enable(kernel_dir=self.kernel_dir_relu, dispatch_key=device.upper())

        torch.testing.assert_close(relu(x), watermarked)

        # Disable monkey patching
        BackendBench.disable()
        torch.testing.assert_close(relu(x), expected)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ],
    )
    def test_context_manager_relu(self, setup_dir_relu, device):
        """Test that context manager enables and disables correctly."""
        BackendBench.disable()
        relu = torch.ops.aten.relu.default
        x = torch.tensor([-1.0, 0.0, 1.0], device=device)
        expected = torch.tensor([0.0, 0.0, 1.0], device=device)
        watermarked = torch.full_like(x, get_operator_watermark_value("relu.default"))

        torch.testing.assert_close(relu(x), expected)

        with BackendBench.BackendBench.enable(
            kernel_dir=self.kernel_dir_relu, dispatch_key=device.upper()
        ):
            torch.testing.assert_close(relu(x), watermarked)

        torch.testing.assert_close(relu(x), expected)

    def test_context_manager_nested_behavior(self, setup_dir_relu):
        """Test context manager behavior when BackendBench is already enabled."""
        BackendBench.disable()
        relu = torch.ops.aten.relu.default
        leaky_relu = torch.ops.aten.leaky_relu.default
        x = torch.tensor([-1.0, 0.0, 1.0])
        expected_relu = torch.tensor([0.0, 0.0, 1.0])
        expected_leaky_relu = torch.tensor([-0.01, 0.0, 1.0])
        watermarked_relu = torch.full_like(x, get_operator_watermark_value("relu.default"))
        watermarked_leaky_relu = torch.full_like(
            x, get_operator_watermark_value("leaky_relu.default")
        )

        BackendBench.enable(kernel_dir=self.kernel_dir_relu, dispatch_key="CPU")

        torch.testing.assert_close(relu(x), watermarked_relu)
        torch.testing.assert_close(leaky_relu(x), expected_leaky_relu)

        with BackendBench.BackendBench.enable(
            kernel_dir=self.kernel_dir_leaky_relu, dispatch_key="CPU"
        ):
            torch.testing.assert_close(relu(x), watermarked_relu)
            torch.testing.assert_close(leaky_relu(x), watermarked_leaky_relu)

        torch.testing.assert_close(relu(x), watermarked_relu)
        torch.testing.assert_close(leaky_relu(x), expected_leaky_relu)

        BackendBench.disable()
        torch.testing.assert_close(relu(x), expected_relu)
        torch.testing.assert_close(leaky_relu(x), expected_leaky_relu)

    def test_context_manager_with_exception(self, setup_dir_relu):
        """Test that context manager properly disables even when exception occurs."""
        BackendBench.disable()
        relu = torch.ops.aten.relu.default
        x = torch.tensor([-1.0, 0.0, 1.0])
        expected = torch.tensor([0.0, 0.0, 1.0])

        torch.testing.assert_close(relu(x), expected)

        try:
            with BackendBench.BackendBench.enable(
                kernel_dir=self.kernel_dir_relu, dispatch_key="CPU"
            ):
                raise ValueError("Test exception")
        except ValueError:
            pass

        torch.testing.assert_close(relu(x), expected)
