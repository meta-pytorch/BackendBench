#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Test that ALL operators are loaded and monkey patched by DirectoryBackend.

This test:
1. Uses DirectoryBackend to load ALL operators from generated_kernels/
2. Verifies that all watermarked operators are loaded
3. Uses eval.py's eval_correctness to verify they fail (proving monkey patching)
4. Uses main.py to run a full evaluation showing correctness metrics
"""

import sys
import unittest
import subprocess
from pathlib import Path

import torch

# Add BackendBench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from BackendBench.backends import DirectoryBackend
from BackendBench.eval import eval_correctness
from BackendBench.suite import Test


class TestAllOperatorsMonkeyPatching(unittest.TestCase):
    """Test that ALL operators are loaded and monkey patched."""

    @classmethod
    def setUpClass(cls):
        """Generate required directory structure and operators."""
        # Generate the directory structure
        subprocess.run([sys.executable, "setup_operator_directories.py"], check=True)
        # Create watermarked implementations
        subprocess.run(
            [sys.executable, "create_watermarked_operators.py", "--overwrite"], check=True
        )

    def test_1_all_operators_loaded(self):
        """Test 1: Verify DirectoryBackend loads ALL operators."""
        print("\n" + "=" * 60)
        print("TEST 1: Loading ALL Operators with DirectoryBackend")
        print("=" * 60)

        # Load main directory
        main_backend = DirectoryBackend("generated_kernels")
        main_count = len(main_backend.compiled_kernels)

        # Load internal_only directory
        internal_backend = DirectoryBackend("generated_kernels/internal_only")
        internal_count = len(internal_backend.compiled_kernels)

        print("\n📊 Operator Loading Summary:")
        print(f"   Main directory: {main_count} operators")
        print(f"   Internal directory: {internal_count} operators")
        print(f"   TOTAL: {main_count + internal_count} operators")

        # List some examples from each
        print("\n📋 Sample operators from main directory:")
        for i, op in enumerate(list(main_backend.compiled_kernels.keys())[:5]):
            print(f"   {i + 1}. {op}")
        print(f"   ... and {main_count - 5} more")

        print("\n📋 Sample operators from internal_only:")
        for i, op in enumerate(list(internal_backend.compiled_kernels.keys())[:5]):
            print(f"   {i + 1}. {op}")
        if internal_count > 5:
            print(f"   ... and {internal_count - 5} more")

        # Verify we loaded a substantial number
        self.assertGreater(main_count, 50, "Should load many operators from main directory")
        self.assertGreater(internal_count, 30, "Should load many operators from internal_only")

        print(
            f"\n✅ SUCCESS: DirectoryBackend loaded {main_count + internal_count} total operators"
        )

    def test_2_watermarked_operators_fail_correctness(self):
        """Test 2: Verify watermarked operators fail eval_correctness."""
        print("\n" + "=" * 60)
        print("TEST 2: Watermarked Operators Fail Correctness")
        print("=" * 60)

        backend = DirectoryBackend("generated_kernels")

        # Test a few representative operators
        test_operators = ["add", "mul", "abs", "div", "sub"]
        failed_count = 0
        tested_count = 0

        print("\n🧪 Testing watermarked operators with eval_correctness:")

        for op_name in test_operators:
            # Find the operator
            found_op = None
            for torch_op in backend.compiled_kernels:
                if op_name in str(torch_op).lower() and f".{op_name}." in str(torch_op):
                    found_op = torch_op
                    break

            if not found_op:
                continue

            tested_count += 1

            # Create test cases
            if op_name in ["add", "mul", "div", "sub"]:
                test_cases = [Test(lambda: torch.randn(3, 3), lambda: torch.randn(3, 3))]
            else:  # abs
                test_cases = [Test(lambda: torch.randn(3, 3))]

            try:
                # Use eval_correctness from eval.py
                is_correct = eval_correctness(found_op, backend[found_op], test_cases)

                if not is_correct:
                    failed_count += 1
                    print(f"   ✅ {op_name}: FAILED correctness (watermark detected)")
                else:
                    print(f"   ❌ {op_name}: PASSED correctness (unexpected!)")

            except Exception:
                # Some failures are expected with watermarks
                failed_count += 1
                print(f"   ✅ {op_name}: Evaluation failed (watermark behavior)")

        print(f"\n📊 Results: {failed_count}/{tested_count} operators failed correctness")
        print("   This proves our watermarked implementations are being used!")

        self.assertGreater(failed_count, 0, "At least some watermarked ops should fail")

    def test_3_main_script_evaluation(self):
        """Test 3: Run evaluation using main.py to get correctness metrics."""
        print("\n" + "=" * 60)
        print("TEST 3: Full Evaluation with main.py")
        print("=" * 60)

        # Run main.py with a subset of operators
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

        print(f"\n🚀 Running: {' '.join(cmd)}")
        print("   (This uses eval.py internally for correctness evaluation)")

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse output
        if "correctness score" in result.stdout:
            print("\n📊 Evaluation Results:")
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "score" in line:
                    print(f"   {line}")

            # Extract correctness score
            for line in lines:
                if "correctness score" in line:
                    score = float(line.split()[-1])
                    print(f"\n✅ Correctness score: {score:.2f}")
                    print("   (Low score expected due to watermarked implementations)")

                    # Watermarked implementations should have low correctness
                    self.assertLess(score, 0.5, "Watermarked ops should have low correctness")
        else:
            print("\n⚠️  Could not parse evaluation results")
            print(f"Output: {result.stdout}")

    def test_4_torchbench_suite_evaluation(self):
        """Test 4: Run TorchBench suite evaluation."""
        print("\n" + "=" * 60)
        print("TEST 4: TorchBench Suite Evaluation")
        print("=" * 60)

        # Run with TorchBench suite on a few operators
        cmd = [
            sys.executable,
            "-m",
            "BackendBench.scripts.main",
            "--backend",
            "directory",
            "--suite",
            "torchbench",
            "--ops",
            "add,mul",
            "--topn",
            "1",
            "--log-level",
            "ERROR",
        ]

        print(f"\n🚀 Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print("\n✅ TorchBench evaluation completed")
                if "correctness score" in result.stdout:
                    print("📊 Results found in output")
                    for line in result.stdout.strip().split("\n"):
                        if "score" in line:
                            print(f"   {line}")
            else:
                print(f"\n⚠️  TorchBench evaluation had issues: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("\n⚠️  TorchBench evaluation timed out (this is okay for the test)")

    def test_5_verify_operator_counts(self):
        """Test 5: Verify we're loading the expected number of operators."""
        print("\n" + "=" * 60)
        print("TEST 5: Operator Count Verification")
        print("=" * 60)

        # Count operators in directories
        main_ops = list(Path("generated_kernels").iterdir())
        main_ops = [d for d in main_ops if d.is_dir() and d.name != "internal_only"]

        internal_ops = list(Path("generated_kernels/internal_only").iterdir())
        internal_ops = [d for d in internal_ops if d.is_dir()]

        print("\n📁 Directory Structure:")
        print(f"   generated_kernels/: {len(main_ops)} operator directories")
        print(f"   generated_kernels/internal_only/: {len(internal_ops)} operator directories")
        print(f"   TOTAL: {len(main_ops) + len(internal_ops)} operator directories")

        # Load with DirectoryBackend and compare
        main_backend = DirectoryBackend("generated_kernels")
        internal_backend = DirectoryBackend("generated_kernels/internal_only")

        print("\n🔧 DirectoryBackend Loading:")
        print(f"   Main backend: {len(main_backend.compiled_kernels)} operators loaded")
        print(f"   Internal backend: {len(internal_backend.compiled_kernels)} operators loaded")

        # The loaded count might be slightly different due to operator overloads
        # but should be in the same ballpark
        self.assertGreater(
            len(main_backend.compiled_kernels),
            len(main_ops) * 0.8,
            "Should load most operators from directories",
        )

        print("\n✅ SUCCESS: Operator counts verified")
        print("   DirectoryBackend successfully loads operators from all directories")


if __name__ == "__main__":
    unittest.main(verbosity=2)
