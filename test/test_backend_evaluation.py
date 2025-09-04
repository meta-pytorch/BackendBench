#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
import subprocess
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from BackendBench.backends import DirectoryBackend
from BackendBench.eval import eval_correctness
from BackendBench.suite import Test


class TestBackendEvaluation(unittest.TestCase):
    """Comprehensive test for backend evaluation system."""

    @classmethod
    def setUpClass(cls):
        """Generate required directory structure and operators."""
        from pathlib import Path

        base_dir = Path("generated_kernels")
        test_ops = ["bitwise_and", "fmod", "relu", "add", "mul", "div"]

        for op_name in test_ops:
            op_dir = base_dir / op_name
            op_dir.mkdir(parents=True, exist_ok=True)

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

    def test_1_directory_backend_loads_operators(self):
        """Verify DirectoryBackend loads operators correctly."""
        backend = DirectoryBackend("generated_kernels")
        operator_count = len(backend.compiled_kernels)

        self.assertGreater(operator_count, 0, "Should load operators from generated_kernels")
        self.assertIsInstance(backend.compiled_kernels, dict)

    def test_2_watermarked_implementations_fail_correctness(self):
        """Verify watermarked operators fail eval_correctness (proving monkey patching)."""
        backend = DirectoryBackend("generated_kernels")

        failed_count = 0
        total_tested = 0

        test_ops = [
            (
                torch.ops.aten.bitwise_and.Tensor,
                lambda: torch.tensor([1, 2, 3]),
                lambda: torch.tensor([2, 3, 4]),
            ),
            (
                torch.ops.aten.fmod.Tensor,
                lambda: torch.tensor([5.0, 7.0]),
                lambda: torch.tensor([2.0, 3.0]),
            ),
        ]

        for op, *arg_generators in test_ops:
            if op in backend:
                impl = backend[op]
                test = Test(*arg_generators)
                correctness, correctness_results = eval_correctness(op, impl, [test])
                assert len(correctness_results) == 1
                total_tested += 1
                if correctness == 0.0:
                    failed_count += 1

        self.assertGreater(total_tested, 0, "Should test at least one operator")
        self.assertGreater(failed_count, 0, "At least some watermarked ops should fail")

    def test_3_main_script_evaluation(self):
        """Verify main.py script works with DirectoryBackend."""
        cmd = [
            sys.executable,
            "-m",
            "BackendBench.scripts.main",
            "--backend",
            "directory",
            "--suite",
            "smoke",
            "--log-level",
            "ERROR",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        self.assertEqual(result.returncode, 0, "Main script should complete successfully")
        self.assertIsInstance(result.stdout, str)
        self.assertIsInstance(result.stderr, str)


if __name__ == "__main__":
    unittest.main()
