# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .base_directory_backend import BaseDirectoryBackendABS

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ..suite.custom_ops import CustomOpsTestSuite


class CustomOpsBackend(BaseDirectoryBackendABS):
    """
    Directory-based backend for non-ATen custom operators.

    Discovers and loads kernel implementations from the custom ops directory structure.
    Each operator directory should contain Python files with kernel implementations
    following the naming pattern: {op_name}_*.py (excluding gen_input.py and reference files).

    The backend registers each implementation as op__impl_name for testing.
    """

    def __init__(self, suite: "CustomOpsTestSuite" = None, ops_dir: str = "custom_ops"):
        self.suite = suite
        super().__init__("custom_ops", ops_dir)

    def load_test_inputs(self, gen_input_path: Path):
        """
        Load test inputs from gen_input.py file.
        
        Returns:
            tuple: (correctness_tests, performance_tests)
        """
        try:
            # Load the gen_input.py module
            spec = importlib.util.spec_from_file_location("gen_input", gen_input_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec from {gen_input_path}")
                return [], []

            gen_input_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen_input_module)

            # Get correctness tests
            correctness_tests = []
            if hasattr(gen_input_module, 'get_correctness_tests'):
                raw_tests = gen_input_module.get_correctness_tests()
                correctness_tests = self._normalize_tests(raw_tests)

            # Get performance tests  
            performance_tests = []
            if hasattr(gen_input_module, 'get_performance_tests'):
                raw_tests = gen_input_module.get_performance_tests()
                performance_tests = self._normalize_tests(raw_tests)

            logger.debug(f"Loaded {len(correctness_tests)} correctness tests and {len(performance_tests)} performance tests from {gen_input_path}")
            return correctness_tests, performance_tests

        except Exception as e:
            logger.error(f"Failed to load test inputs from {gen_input_path}: {e}")
            return [], []

    def _normalize_tests(self, raw_tests):
        """
        Normalize test inputs to ensure they are all Test objects.
        
        Handles both Test objects and raw tuples/lists that need to be wrapped.
        """
        from ..suite.base import Test

        normalized_tests = []
        for test in raw_tests:
            assert isinstance(test, Test), f"Test must be a Test object, got {type(test)}"
            normalized_tests.append(test)
            # todo: if we want to support more test in gen_input.py

        return normalized_tests

    def load_op_implementations(self, op_name: str, op_dir: Path) -> list[str]:
        """
        Load all kernel implementations from the operations directory.
        """

        inputs = op_dir / "gen_input.py"
        if not inputs.exists():
            logger.warning(f"No gen_input.py found in {op_dir}, skipping {op_name}")
            return []

        # Load test inputs from gen_input.py
        correctness_tests, performance_tests = self.load_test_inputs(inputs)
        
        if not correctness_tests and not performance_tests:
            logger.warning(f"No test inputs found in {inputs}, skipping {op_name}")
            return []

        impl_files = [
            f
            for f in op_dir.iterdir()
            if f.is_file() and f.suffix == ".py" and f.name != "gen_input.py"
        ]

        if not impl_files:
            logger.warning(f"No implementation files found in {op_dir}")
            return []

        ref_impl = None
        for f in impl_files:
            if f.name.startswith(f"{op_name}_reference"):
                ref_impl = f
                # we won't iter more so it's safe to remove it
                impl_files.remove(f)
                break
        else:
            logger.warning(
                f"No reference implementation found for {op_name}, using first implementation as reference"
            )
            ref_impl = impl_files[0]

        ref_impl_kernel = self.load_py_kernel_from_file(ref_impl, op_name)

        # For custom backend, each implementation is consider a seperate op to be tested.
        # Create op for each implementation and register them
        for impl in impl_files:
            impl_name = impl.stem
            impl_kernel = self.load_py_kernel_from_file(impl, op_name)

            torch_op_name = f'{op_name}::{impl_name}'
            ref_impl_as_op = torch.library.custom_op(torch_op_name, ref_impl_kernel, mutates_args=(), device_types='cuda')
            impl_as_op = torch.library.custom_op(torch_op_name, impl_kernel, mutates_args=())

            setattr(ref_impl_as_op, '__name__', torch_op_name)
            setattr(impl_as_op, '__name__', torch_op_name)

            # Store the wrapped kernel with its unique name
            self.compiled_kernels[ref_impl_as_op] = impl_as_op

            # Use the loaded test inputs instead of empty lists
            # Pass the wrapped kernel function
            self.suite.add_test(op_name, ref_impl_as_op, correctness_tests, performance_tests)

        return list(self.compiled_kernels.keys())
