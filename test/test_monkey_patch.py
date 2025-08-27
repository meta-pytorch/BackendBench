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

# Add BackendBench to path

import BackendBench
from BackendBench.scripts.create_watermarked_operators import get_operator_watermark_value


class TestMonkeyPatch:
    """Comprehensive test for backend evaluation system."""

    def setUpDirReLU(self):
        """Generate required directory structure and operators."""
        # Generate the directory structure
        subprocess.run(
            [sys.executable, "-m", "BackendBench.scripts.setup_operator_directories"], check=True
        )
        # Create watermarked implementations with unique values to catch cross-contamination
        subprocess.run(
            [
                sys.executable,
                "-m",
                "BackendBench.scripts.create_watermarked_operators",
                "--overwrite",
                "--unique-watermarks",
            ],
            check=True,
        )

        kernel_dir = "generated_kernels"
        for directory in os.listdir(kernel_dir):
            if directory != "relu" and os.path.isdir(os.path.join(kernel_dir, directory)):
                shutil.rmtree(os.path.join(kernel_dir, directory))

    def test_monkey_patch_relu(self):
        self.setUpDirReLU()

        relu = torch.ops.aten.relu.default
        x = torch.tensor([-1.0, 0.0, 1.0])
        expected = torch.tensor([0.0, 0.0, 1.0])
        watermarked = torch.full_like(x, get_operator_watermark_value("relu"))

        torch.testing.assert_close(relu(x), expected)

        # Enable monkey patching
        BackendBench.enable()
        torch.testing.assert_close(relu(x), watermarked)

        # Disable monkey patching
        BackendBench.disable()
        torch.testing.assert_close(relu(x), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
