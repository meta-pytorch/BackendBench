# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unittest
from BackendBench.eval import eval_one_op, eval_correctness, eval_performance
from BackendBench.suite import Test

def randn(*args, **kwargs):
    return lambda: torch.randn(*args, **kwargs)

class TestVerboseMode(unittest.TestCase):
    def test_verbose_data_structure(self):
        """Test that verbose mode collects correct data structure."""
        
        # Create a simple operation and implementation
        def reference_op(x):
            return x.relu()
        
        def test_impl(x):
            return torch.nn.functional.relu(x)
        
        # Create test cases
        correctness_tests = [
            Test(randn(2, 3, device="cpu")),
            Test(randn(4, 5, device="cpu")),
        ]
        performance_tests = [
            Test(randn(100, 100, device="cpu")),
        ]
        
        # Run evaluation with verbose mode
        verbose_data = {}
        eval_correctness(reference_op, test_impl, correctness_tests, verbose_data)
        eval_performance(reference_op, test_impl, performance_tests, verbose_data)
        
        # Check verbose data structure
        self.assertGreater(len(verbose_data), 0, "Verbose data should not be empty")
        
        for args_str, data in verbose_data.items():
            # Check required keys exist
            self.assertIn("correctness_score", data)
            self.assertIn("correctness_errors", data)
            self.assertIn("absolute_error", data)
            self.assertIn("relative_error", data)
            
            # Check types
            self.assertIn(data["correctness_score"], [0, 1])
            self.assertIsInstance(data["correctness_errors"], str)
            
            # Performance tests should have additional fields
            if "benchmark_time" in data:
                self.assertIn("speedup", data)
                self.assertIsInstance(data["benchmark_time"], (float, int))
                self.assertIsInstance(data["speedup"], (float, int))
    
    def test_verbose_with_eval_one_op(self):
        """Test verbose mode through eval_one_op."""
        
        def reference_op(x):
            return x.abs()
        
        def test_impl(x):
            return torch.abs(x)
        
        correctness_tests = [Test(randn(10, device="cpu"))]
        performance_tests = [Test(randn(1000, device="cpu"))]
        
        verbose_data = {}
        correctness, performance = eval_one_op(
            reference_op, test_impl, correctness_tests, performance_tests, verbose_data
        )
        
        # Should have data for unique arg combinations
        self.assertGreater(len(verbose_data), 0)
        
        # Correctness should be perfect for this simple case
        self.assertEqual(correctness, 1.0)
        
        # Check all entries have complete data
        for data in verbose_data.values():
            self.assertIn("correctness_score", data)
            self.assertEqual(data["correctness_score"], 1)  # Should pass
            
            # Performance test entries should have benchmark_time and speedup
            if "benchmark_time" in data:
                self.assertIn("speedup", data)
    
    def test_verbose_with_failing_impl(self):
        """Test verbose mode with a failing implementation."""
        
        def reference_op(x):
            return x * 2
        
        def failing_impl(x):
            raise RuntimeError("Intentional failure")
        
        tests = [Test(randn(5, device="cpu"))]
        
        verbose_data = {}
        correctness = eval_correctness(reference_op, failing_impl, tests, verbose_data)
        
        self.assertEqual(correctness, 0.0)
        self.assertEqual(len(verbose_data), 1)
        
        # Check error is recorded
        data = list(verbose_data.values())[0]
        self.assertEqual(data["correctness_score"], 0)
        self.assertIn("Intentional failure", data["correctness_errors"])
        self.assertEqual(data["absolute_error"], "")
        self.assertEqual(data["relative_error"], "")

if __name__ == "__main__":
    unittest.main()