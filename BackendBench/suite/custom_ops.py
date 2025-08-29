# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import torch

from .base import Test, OpTest, TestSuite


def randn(*args, **kwargs):
    return lambda: torch.randn(*args, **kwargs)


def tensor(*args, **kwargs):
    return lambda: torch.tensor(*args, **kwargs)


# Define custom operations
def copy_op(input):
    """Reference implementation for copy operation."""
    return input.clone()


def add_optimized_op(input, other, *, alpha=1):
    """Reference implementation for optimized add operation."""
    if isinstance(other, (int, float)):
        return input + alpha * other
    
    if alpha != 1:
        other = other * alpha
    
    return input + other


CustomOpsTestSuite = TestSuite(
    "custom_ops",
    [
        # Copy operation tests
        OpTest(
            copy_op,
            [
                Test(tensor([1.0, 2.0, 3.0, 4.0, 5.0])),
                Test(tensor([[1.0, 2.0], [3.0, 4.0]])),
                Test(randn(10, 10)),
            ],
            [
                Test(randn(1000, 1000)),  # Performance test with large tensor
                Test(randn(10000)),       # Performance test with large 1D tensor
            ],
        ),
        # Add optimized operation tests
        OpTest(
            add_optimized_op,
            [
                Test(tensor([1.0, 2.0, 3.0]), tensor([4.0, 5.0, 6.0])),
                Test(tensor([1.0, 2.0, 3.0]), lambda: 2.0),
                Test(tensor([[1.0, 2.0], [3.0, 4.0]]), tensor([1.0, 2.0])),
                Test(tensor([1.0, 2.0, 3.0]), tensor([1.0, 1.0, 1.0]), alpha=lambda: 2.0),
            ],
            [
                Test(randn(1000, 1000), randn(1000, 1000)),
                Test(randn(10000), randn(10000)),
                Test(randn(100, 100), lambda: 1.5),
            ],
        ),
    ],
)
