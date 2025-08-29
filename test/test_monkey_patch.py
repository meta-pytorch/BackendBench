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
import sys
import shutil
import subprocess
import pytest

import torch

import BackendBench
from BackendBench.scripts.create_watermarked_operators import get_operator_watermark_value


class TestMonkeyPatch:
    kernel_dir = "generated_kernels_test_monkey_path"

    @pytest.fixture(scope="module")
    def setup_dir_relu(self):
        """Generate required directory structure and operators."""
        # Generate the directory structure
        subprocess.run(
            [
                sys.executable,
                "-m",
                "BackendBench.scripts.setup_operator_directories",
                "--base-dir",
                self.kernel_dir,
            ],
            check=True,
        )
        # Create watermarked implementations with unique values to catch cross-contamination
        subprocess.run(
            [
                sys.executable,
                "-m",
                "BackendBench.scripts.create_watermarked_operators",
                "--base-dir",
                self.kernel_dir,
                "--overwrite",
                "--unique-watermarks",
            ],
            check=True,
        )
        for directory in os.listdir(self.kernel_dir):
            if directory != "relu" and os.path.isdir(os.path.join(self.kernel_dir, directory)):
                shutil.rmtree(os.path.join(self.kernel_dir, directory))

        yield

        # Teardown
        shutil.rmtree(self.kernel_dir)

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
        watermarked = torch.full_like(x, get_operator_watermark_value("relu"))

        torch.testing.assert_close(relu(x), expected)

        # Enable monkey patching
        BackendBench.enable(kernel_dir=self.kernel_dir, dispatch_key=device.upper())
        torch.testing.assert_close(relu(x), watermarked)

        # Disable monkey patching
        BackendBench.disable()
        torch.testing.assert_close(relu(x), expected)
