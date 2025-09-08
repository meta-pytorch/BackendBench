# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import subprocess
import sys
from pathlib import Path

from BackendBench.backends.custom_ops import CustomOpsBackend
from BackendBench.suite.custom_ops import CustomOpsTestSuite


class TestCustomOps:
    """Test custom ops functionality."""

    def setup_method(self):
        """Set up test environment using the actual test/custom_ops directory."""
        self.custom_ops_dir = Path(__file__).parent / "custom_ops"

    def test_implementations_discovery(self):
        """Test that 3 implementations can be correctly discovered."""
        suite = CustomOpsTestSuite(str(self.custom_ops_dir))
        backend = CustomOpsBackend(suite, str(self.custom_ops_dir))

        # Should discover 3 implementations: myop_py, myop_py_wrong, myop_triton
        assert len(suite.optests) == 3, "Should discover 3 implementations"

        # Verify that all implementations are loaded
        impl_names = []
        for optest in suite.optests:
            impl_name = str(optest.op).split("::")[-1].rstrip(">")
            impl_names.append(impl_name)
            # Each implementation should have a stored kernel
            assert optest.op in backend.compiled_kernels, (
                f"Implementation {impl_name} should have a stored kernel"
            )

        # Verify we have the expected implementations
        expected_impls = ["myop_py", "myop_py_wrong", "myop_triton"]
        for expected in expected_impls:
            assert any(expected in impl for impl in impl_names), (
                f"Should discover {expected} implementation"
            )

    def test_correctness_results(self):
        """Test that correctness test results are as expected (one should fail)."""
        suite = CustomOpsTestSuite(str(self.custom_ops_dir))
        backend = CustomOpsBackend(suite, str(self.custom_ops_dir))

        test_input = torch.ones(8, device="cuda")
        expected_correct = test_input * 2.0
        expected_wrong = torch.zeros_like(test_input)

        correct_count = 0
        wrong_count = 0

        for optest in suite.optests:
            impl_name = str(optest.op).split("::")[-1].rstrip(">")

            # Test the stored kernel (actual implementation)
            stored_kernel = backend.compiled_kernels[optest.op]
            result = stored_kernel(test_input, alpha=2.0)

            if "wrong" in impl_name:
                # Wrong implementation should return zeros
                is_correct = torch.allclose(result, expected_wrong)
                if is_correct:
                    wrong_count += 1
            else:
                # Correct implementations should return input * alpha
                is_correct = torch.allclose(result, expected_correct)
                if is_correct:
                    correct_count += 1

        # Should have 2 correct implementations and 1 wrong implementation
        assert correct_count == 2, f"Should have 2 correct implementations, got {correct_count}"
        assert wrong_count == 1, f"Should have 1 wrong implementation, got {wrong_count}"

    def test_integration_main_script(self):
        """Test custom ops integration using the main script."""
        cmd = [
            sys.executable,
            "-m",
            "BackendBench.scripts.main",
            "--suite",
            "custom_ops",
            "--backend",
            "custom_ops",
            "--custom-ops-root",
            "test/custom_ops",
            "--log-level",
            "ERROR",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # The script should complete successfully
        assert result.returncode == 0, (
            f"Main script should complete successfully. stderr: {result.stderr}"
        )

        # Parse the output to verify results
        output_lines = result.stdout.split("\n")

        # Look for the correctness score in the output
        correctness_score_found = False
        correctness_score = None

        for line in output_lines:
            if "correctness score" in line.lower():
                correctness_score_found = True
                # Extract the score (should be 0.67 for 2/3 correct)
                parts = line.split(":")
                if len(parts) > 1:
                    score_str = parts[1].strip()
                    correctness_score = float(score_str)
                break

        # Verify we found the correctness score
        assert correctness_score_found, "Should find correctness score in output"
        assert correctness_score is not None, "Should be able to parse correctness score"

        # The correctness score should be 0.67 (2 out of 3 implementations are correct)
        expected_score = 2.0 / 3.0  # 0.666...
        assert abs(correctness_score - expected_score) < 0.01, (
            f"Expected correctness score ~{expected_score:.2f}, got {correctness_score}"
        )

        # Verify the output contains the expected summary
        summary_found = False
        for line in output_lines:
            if "Results saved to directory:" in line:
                summary_found = True
                break

        assert summary_found, "Should find results summary in output"
