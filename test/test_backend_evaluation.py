#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive test for BackendBench evaluation system.

Tests:
1. DirectoryBackend loads operators correctly
2. Watermarked implementations fail correctness (proving monkey patching works)
3. Main script evaluation works end-to-end
4. eval.py integration works properly
"""

import sys
import unittest
import subprocess
from pathlib import Path

import torch

# Add BackendBench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from BackendBench.backends import DirectoryBackend
from BackendBench.eval import eval_correctness, eval_one_op
from BackendBench.suite import Test


class TestBackendEvaluation(unittest.TestCase):
    """Comprehensive test for backend evaluation system."""

    @classmethod
    def setUpClass(cls):
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

    def test_1_directory_backend_loads_operators(self):
        """Test 1: Verify DirectoryBackend loads operators correctly."""
        print("\n" + "=" * 60)
        print("TEST 1: DirectoryBackend Operator Loading")
        print("=" * 60)

        backend = DirectoryBackend("generated_kernels")
        operator_count = len(backend.compiled_kernels)

        print(f"\n📊 Loaded {operator_count} operators")

        # List some examples
        print("\n📋 Sample operators:")
        for i, op in enumerate(list(backend.compiled_kernels.keys())[:5]):
            print(f"   {i + 1}. {op}")
        print(f"   ... and {operator_count - 5} more")

        # Verify we loaded a substantial number
        self.assertGreater(operator_count, 100, "Should load many operators from generated_kernels")

        print(f"\n✅ SUCCESS: DirectoryBackend loaded {operator_count} total operators")

    def test_2_watermarked_implementations_fail_correctness(self):
        """Test 2: Verify watermarked operators fail eval_correctness (proving monkey patching)."""
        print("\n" + "=" * 60)
        print("TEST 2: Watermarked Implementation Correctness")
        print("=" * 60)

        backend = DirectoryBackend("generated_kernels")

        print("\n🧪 Testing watermarked operators with eval_correctness:")

        failed_count = 0
        total_tested = 0

        # Test several operators that should have watermarked implementations
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
                try:
                    impl = backend[op]
                    test = Test(*arg_generators)
                    correctness = eval_correctness(op, impl, [test])

                    total_tested += 1
                    if correctness == 0.0:
                        failed_count += 1
                        print(f"  ✓ {str(op).split('.')[-2]}: Failed correctness (watermarked)")
                    else:
                        print(f"  ✗ {str(op).split('.')[-2]}: Passed correctness unexpectedly")

                except Exception as e:
                    print(f"  ? {str(op).split('.')[-2]}: Error testing - {e}")

        print(f"\n📊 Results: {failed_count}/{total_tested} operators failed correctness")
        print("   This proves our watermarked implementations are being used!")

        self.assertGreater(failed_count, 0, "At least some watermarked ops should fail")

    def test_3_main_script_evaluation(self):
        """Test 3: Verify main.py script works with DirectoryBackend."""
        print("\n" + "=" * 60)
        print("TEST 3: Main Script Evaluation")
        print("=" * 60)

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

        print("\n🚀 Running: " + " ".join(cmd))
        print("   (This uses eval.py internally for correctness evaluation)")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        print("\n📊 Evaluation Results:")
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "score" in line:
                    print(f"   {line}")

        # Should complete without crashing
        self.assertEqual(result.returncode, 0, "Main script should complete successfully")

        print("\n✅ SUCCESS: Main script evaluation completed")

    def test_4_eval_integration(self):
        """Test 4: Verify eval.py functions work correctly."""
        print("\n" + "=" * 60)
        print("TEST 4: eval.py Integration")
        print("=" * 60)

        backend = DirectoryBackend("generated_kernels")

        print("\n🔧 Testing eval_one_op function:")

        # Find a watermarked operator to test
        test_op = None
        for op in backend.compiled_kernels.keys():
            if "bitwise_and" in str(op) and "Tensor" in str(op):
                test_op = op
                break

        if test_op:
            impl = backend[test_op]
            test = Test(lambda: torch.tensor([1, 2, 3]), lambda: torch.tensor([2, 3, 4]))

            correctness, performance = eval_one_op(test_op, impl, [test], [test])

            print(f"   Operation: {test_op}")
            print(f"   Correctness: {correctness}")
            print(f"   Performance: {performance}")

            # Watermarked implementation should fail correctness
            self.assertEqual(correctness, 0.0, "Watermarked implementation should fail correctness")

            print("   ✓ eval_one_op works correctly with watermarked implementation")
        else:
            print("   ! No suitable test operator found, skipping detailed test")

        print("\n✅ SUCCESS: eval.py integration verified")


if __name__ == "__main__":
    unittest.main()
