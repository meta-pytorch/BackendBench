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
from pathlib import Path

import torch

# Add BackendBench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import BackendBench
from BackendBench.scripts.create_watermarked_operators import WATERMARK_BASE


class TestMonkeyPatch:
    """Comprehensive test for backend evaluation system."""

    def setUpDir(self):
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
            if directory != "add" and os.path.isdir(os.path.join(kernel_dir, directory)):
                shutil.rmtree(os.path.join(kernel_dir, directory))

    def test_monkey_patch_add(self):
        self.setUpDir()

        a = torch.zeros(3, 3)
        b = torch.ones(3, 3)

        torch.testing.assert_close(a + b, torch.ones_like(a))

        # Enable monkey patching
        BackendBench.enable()

        torch.testing.assert_close(a + b, torch.full_like(a, WATERMARK_BASE))

        # Disable monkey patching
        BackendBench.disable()

        torch.testing.assert_close(a + b, torch.ones_like(a))
