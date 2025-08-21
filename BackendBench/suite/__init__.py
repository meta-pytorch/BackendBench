# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
BackendBench suites submodule.

This module provides various test suite implementations for benchmarking
PyTorch operations across different backends. Each test suite defines a
collection of tests to evaluate the correctness and/or performacne of
backend implementations by comparing them against PyTorch operations.
"""

from .base import Test, OpTest, TestSuite
from .facto import FactoTestSuite
from .opinfo import OpInfoTestSuite
from .smoke import SmokeTestSuite, randn
from .torchbench import TorchBenchOpTest, TorchBenchTestSuite, DEFAULT_HUGGINGFACE_URL

__all__ = [
    "Test",
    "OpTest",
    "TestSuite",
    "FactoTestSuite",
    "OpInfoTestSuite",
    "SmokeTestSuite",
    "randn",
    "TorchBenchOpTest",
    "TorchBenchTestSuite",
    "DEFAULT_HUGGINGFACE_URL",
]
