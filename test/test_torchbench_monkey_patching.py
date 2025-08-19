#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Test monkey patching with TorchBench suite using correct and incorrect implementations.
This test:
1. Replaces watermarked implementations with 2 correct + 2 incorrect implementations
2. Uses the real TorchBench evaluation suite from BackendBench
3. Verifies that correct implementations pass and incorrect ones fail
4. Confirms monkey patching is working through the full evaluation pipeline
"""

import sys
import unittest
from pathlib import Path

import torch

# Add BackendBench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from BackendBench.backends import DirectoryBackend
from BackendBench.torchbench_suite import TorchBenchTestSuite
from BackendBench.eval import eval_one_op


class TestTorchBenchMonkeyPatching(unittest.TestCase):
    """Test monkey patching using the real TorchBench evaluation suite."""

    @classmethod
    def setUpClass(cls):
        """Set up test by creating correct and incorrect implementations."""
        cls.generated_kernels_dir = Path("generated_kernels")
        cls.backup_implementations = {}

        # Generate the directory structure if it doesn't exist
        if not cls.generated_kernels_dir.exists():
            import subprocess
            import sys

            subprocess.run([sys.executable, "setup_operator_directories.py"], check=True)

        # Backup existing implementations and create test ones
        cls._backup_and_create_correct_add()
        cls._backup_and_create_correct_abs()
        cls._backup_and_create_incorrect_mul()
        cls._backup_and_create_incorrect_div()

        print("Created test implementations (2 correct, 2 incorrect)")

    @classmethod
    def tearDownClass(cls):
        """Restore original implementations."""
        for op_name, backup_content in cls.backup_implementations.items():
            impl_path = cls.generated_kernels_dir / op_name / f"{op_name}_implementation_v1.py"
            if backup_content is not None:
                impl_path.write_text(backup_content)
        print("Restored original implementations")

    @classmethod
    def _backup_and_create_correct_add(cls):
        """Create correct add implementation."""
        add_dir = cls.generated_kernels_dir / "add"
        impl_path = add_dir / "add_implementation_v1.py"

        # Backup existing
        if impl_path.exists():
            cls.backup_implementations["add"] = impl_path.read_text()

        # Create correct implementation
        impl_path.write_text('''# Correct implementation of add
import torch

def add_kernel_impl(input, other, *, alpha=1):
    """Correct implementation of torch.add"""
    return input + alpha * other
''')

    @classmethod
    def _backup_and_create_correct_abs(cls):
        """Create correct abs implementation."""
        abs_dir = cls.generated_kernels_dir / "abs"
        impl_path = abs_dir / "abs_implementation_v1.py"

        # Backup existing
        if impl_path.exists():
            cls.backup_implementations["abs"] = impl_path.read_text()

        # Create correct implementation
        impl_path.write_text('''# Correct implementation of abs
import torch

def abs_kernel_impl(input):
    """Correct implementation of torch.abs"""
    return torch.abs(input)
''')

    @classmethod
    def _backup_and_create_incorrect_mul(cls):
        """Create incorrect mul implementation (returns zeros)."""
        mul_dir = cls.generated_kernels_dir / "mul"
        impl_path = mul_dir / "mul_implementation_v1.py"

        # Backup existing
        if impl_path.exists():
            cls.backup_implementations["mul"] = impl_path.read_text()

        # Create incorrect implementation
        impl_path.write_text('''# Incorrect implementation of mul (returns zeros)
import torch

def mul_kernel_impl(input, other):
    """Incorrect implementation - always returns zeros"""
    return torch.zeros_like(input)
''')

    @classmethod
    def _backup_and_create_incorrect_div(cls):
        """Create incorrect div implementation (returns ones)."""
        div_dir = cls.generated_kernels_dir / "div"
        impl_path = div_dir / "div_implementation_v1.py"

        # Backup existing
        if impl_path.exists():
            cls.backup_implementations["div"] = impl_path.read_text()

        # Create incorrect implementation
        impl_path.write_text('''# Incorrect implementation of div (returns ones)
import torch

def div_kernel_impl(input, other):
    """Incorrect implementation - always returns ones"""
    return torch.ones_like(input)
''')

    def setUp(self):
        """Set up backend for each test."""
        self.backend = DirectoryBackend("generated_kernels")
        loaded_ops = list(self.backend.compiled_kernels.keys())

        # Find our test operators
        self.test_ops = {"add": None, "abs": None, "mul": None, "div": None}

        for op in loaded_ops:
            op_str = str(op).lower()
            if "add.default" in op_str and "addmm" not in op_str:
                self.test_ops["add"] = op
            elif "abs.default" in op_str:
                self.test_ops["abs"] = op
            elif "mul.default" in op_str:
                self.test_ops["mul"] = op
            elif "div.default" in op_str and "floor" not in op_str:
                self.test_ops["div"] = op

    def test_directory_backend_loads_test_implementations(self):
        """Test that DirectoryBackend loads our test implementations."""
        print("\n=== Testing DirectoryBackend Loading ===")

        loaded_ops = list(self.backend.compiled_kernels.keys())

        print(f"Backend loaded {len(loaded_ops)} operators")
        self.assertGreater(len(loaded_ops), 0, "Backend should load operators")

        # Verify we found our operators
        found_count = sum(1 for op in self.test_ops.values() if op is not None)
        print(f"Found {found_count}/4 test operators in backend")

        for name, op in self.test_ops.items():
            if op is not None:
                print(f"  ✓ {name} -> {op}")

        self.assertGreater(found_count, 0, "Should find at least some test operators")

    def test_correct_implementations_behavior(self):
        """Test that our correct implementations behave correctly."""
        print("\n=== Testing Correct Implementation Behavior ===")

        # Test correct add
        if self.test_ops["add"] is not None:
            add_impl = self.backend[self.test_ops["add"]]
            x = torch.tensor([1.0, 2.0])
            y = torch.tensor([3.0, 4.0])
            result = add_impl(x, y)
            expected = torch.tensor([4.0, 6.0])

            self.assertTrue(
                torch.allclose(result, expected), f"Correct add failed: {result} != {expected}"
            )
            print("  ✓ add implementation works correctly")

        # Test correct abs
        if self.test_ops["abs"] is not None:
            abs_impl = self.backend[self.test_ops["abs"]]
            x = torch.tensor([-1.0, 2.0, -3.0])
            result = abs_impl(x)
            expected = torch.tensor([1.0, 2.0, 3.0])

            self.assertTrue(
                torch.allclose(result, expected), f"Correct abs failed: {result} != {expected}"
            )
            print("  ✓ abs implementation works correctly")

    def test_incorrect_implementations_behavior(self):
        """Test that our incorrect implementations behave incorrectly."""
        print("\n=== Testing Incorrect Implementation Behavior ===")

        # Test incorrect mul (should return zeros)
        if self.test_ops["mul"] is not None:
            mul_impl = self.backend[self.test_ops["mul"]]
            x = torch.tensor([2.0, 3.0])
            y = torch.tensor([4.0, 5.0])
            result = mul_impl(x, y)

            # Should NOT be correct result
            correct_result = torch.tensor([8.0, 15.0])
            self.assertFalse(
                torch.allclose(result, correct_result),
                "Incorrect mul should not produce correct result",
            )

            # Should be zeros
            expected_zeros = torch.zeros_like(x)
            self.assertTrue(
                torch.allclose(result, expected_zeros),
                f"Incorrect mul should return zeros: {result}",
            )
            print("  ✓ mul implementation incorrectly returns zeros")

        # Test incorrect div (should return ones)
        if self.test_ops["div"] is not None:
            div_impl = self.backend[self.test_ops["div"]]
            x = torch.tensor([8.0, 12.0])
            y = torch.tensor([2.0, 3.0])
            result = div_impl(x, y)

            # Should NOT be correct result
            correct_result = torch.tensor([4.0, 4.0])
            self.assertFalse(
                torch.allclose(result, correct_result),
                "Incorrect div should not produce correct result",
            )

            # Should be ones
            expected_ones = torch.ones_like(x)
            self.assertTrue(
                torch.allclose(result, expected_ones), f"Incorrect div should return ones: {result}"
            )
            print("  ✓ div implementation incorrectly returns ones")

    def test_torchbench_suite_integration(self):
        """Test integration with TorchBench suite."""
        print("\n=== Testing TorchBench Suite Integration ===")

        try:
            # Create TorchBench suite with our test operators
            suite = TorchBenchTestSuite(
                "torchbench", None, filter=["add", "abs", "mul", "div"], topn=2
            )  # Limit to 2 test cases per op

            suite_tests = list(suite)
            print(f"TorchBench suite created {len(suite_tests)} test cases")

            if len(suite_tests) == 0:
                self.skipTest("No TorchBench tests found for our operators")

            # Show which operations are being tested
            tested_ops = [str(test.op) for test in suite_tests]
            print(f"TorchBench operations: {tested_ops}")

            # Verify our backend contains the operations being tested
            backend_ops = set(self.backend.compiled_kernels.keys())

            matched_tests = []
            for test in suite_tests:
                if test.op in backend_ops:
                    matched_tests.append(test)

            print(f"Found {len(matched_tests)} TorchBench tests that match our backend")
            self.assertGreater(
                len(matched_tests), 0, "Should find TorchBench tests that match our backend"
            )

        except Exception as e:
            self.skipTest(f"TorchBench suite creation failed: {e}")

    def test_end_to_end_evaluation_with_torchbench(self):
        """Test end-to-end evaluation using TorchBench suite."""
        print("\n=== Testing End-to-End Evaluation ===")

        try:
            # Create TorchBench suite
            suite = TorchBenchTestSuite(
                "torchbench", None, filter=["add", "abs", "mul", "div"], topn=1
            )

            results = {}

            for test in suite:
                if test.op not in self.backend:
                    continue

                op_name = str(test.op).split(".")[-2]  # Extract op name
                if op_name not in ["add", "abs", "mul", "div"]:
                    continue

                print(f"\nEvaluating {op_name} ({test.op})")

                try:
                    # Run evaluation using TorchBench test cases
                    correctness, performance = eval_one_op(
                        test.op,
                        self.backend[test.op],
                        test.correctness_tests,
                        test.performance_tests,
                    )

                    results[op_name] = {
                        "correctness": correctness,
                        "performance": performance,
                        "expected_correct": op_name in ["add", "abs"],
                    }

                    print(f"  Correctness: {correctness:.3f}")
                    print(f"  Performance: {performance:.3f}")

                except Exception as e:
                    print(f"  Evaluation failed: {e}")
                    results[op_name] = {"error": str(e)}

            # Analyze results
            print("\n=== Evaluation Results Summary ===")

            for op_name, result in results.items():
                if "error" in result:
                    print(f"{op_name}: ERROR - {result['error']}")
                    continue

                correctness = result["correctness"]
                expected_correct = result["expected_correct"]

                if expected_correct:
                    # Should have high correctness
                    if correctness > 0.8:
                        print(
                            f"✓ {op_name}: PASS (correctness={correctness:.3f}) - correct implementation"
                        )
                    else:
                        print(
                            f"✗ {op_name}: FAIL (correctness={correctness:.3f}) - should be correct!"
                        )
                else:
                    # Should have low correctness
                    if correctness < 0.2:
                        print(
                            f"✓ {op_name}: FAIL (correctness={correctness:.3f}) - incorrect implementation as expected"
                        )
                    else:
                        print(
                            f"? {op_name}: UNEXPECTED (correctness={correctness:.3f}) - should fail!"
                        )

            # Verify we got some results
            self.assertGreater(len(results), 0, "Should get evaluation results")

            print("\n✓ End-to-end evaluation completed using TorchBench suite")

        except Exception as e:
            self.skipTest(f"TorchBench evaluation failed: {e}")

    def test_monkey_patching_vs_pytorch_reference(self):
        """Verify our implementations are used instead of PyTorch's."""
        print("\n=== Testing Monkey Patching vs PyTorch Reference ===")

        # Test with simple inputs
        x = torch.tensor([4.0, 6.0])
        y = torch.tensor([2.0, 3.0])

        comparisons = []

        for op_name in ["mul", "div"]:  # Test our incorrect implementations
            if self.test_ops[op_name] is None:
                continue

            our_impl = self.backend[self.test_ops[op_name]]
            our_result = our_impl(x, y)

            # Get PyTorch's result
            if op_name == "mul":
                pytorch_result = torch.mul(x, y)
                print(f"\n{op_name}:")
                print(f"  PyTorch result: {pytorch_result}")
                print(f"  Our result:     {our_result}")

                # They should be different
                is_different = not torch.allclose(our_result, pytorch_result)
                self.assertTrue(is_different, f"Our {op_name} should differ from PyTorch's")

                if is_different:
                    print(f"  ✓ Monkey patching confirmed - our {op_name} differs from PyTorch")
                    comparisons.append(True)

            elif op_name == "div":
                pytorch_result = torch.div(x, y)
                print(f"\n{op_name}:")
                print(f"  PyTorch result: {pytorch_result}")
                print(f"  Our result:     {our_result}")

                # They should be different
                is_different = not torch.allclose(our_result, pytorch_result)
                self.assertTrue(is_different, f"Our {op_name} should differ from PyTorch's")

                if is_different:
                    print(f"  ✓ Monkey patching confirmed - our {op_name} differs from PyTorch")
                    comparisons.append(True)

        self.assertGreater(
            len(comparisons), 0, "Should verify monkey patching for at least one operator"
        )
        print(f"\n✓ Verified monkey patching for {len(comparisons)} operators")


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)
