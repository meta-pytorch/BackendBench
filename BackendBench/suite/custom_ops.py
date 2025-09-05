# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
from pathlib import Path
from typing import List, Optional, Callable

from BackendBench.backends.custom_ops import CustomOpsBackend

from .base import Test, OpTest, TestSuite

logger = logging.getLogger(__name__)


def _load_module_from_path(mod_name: str, file_path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_tests(raw_tests) -> List[Test]:
    """Convert raw test data to Test objects."""
    if raw_tests is None:
        return []
    
    tests: List[Test] = []
    for item in raw_tests:
        if isinstance(item, Test):
            tests.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
            # (args, kwargs) format
            args, kwargs = item
            # Ensure args is a tuple of callables
            if isinstance(args, (list, tuple)):
                args_tuple = tuple(args)
            else:
                args_tuple = (args,)
            tests.append(Test(*args_tuple, **kwargs))
        else:
            # Single argument or tuple of arguments
            if isinstance(item, (list, tuple)):
                tests.append(Test(*item))
            else:
                tests.append(Test(item))
    return tests


class CustomOpsTestSuite(TestSuite):
    """
    Filter custom ops by op name and impl name
    """

    def __init__(self, root_dir: str = "custom_ops", filter=None):
        self.root = Path(root_dir)
        self.filter = filter
        super().__init__(            'custom_ops', []        )

    def __iter__(self):
        for optest in self.optests:
            if self.filter is not None:
                if optest.op.__name__ not in self.filter or self.op_name_map[optest.op] not in self.filter:
                    continue
            yield optest

    op_name_map = {}
    def add_test(self, op_name, impl_as_op, correctness_tests, performance_tests):
        self.op_name_map[impl_as_op] = op_name
        self.optests.append(OpTest(impl_as_op, correctness_tests, performance_tests))
