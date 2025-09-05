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


class CustomOpsTestSuite(TestSuite):
    """
    Filter custom ops by op name and impl name
    """

    def __init__(self, root_dir: str = "custom_ops", filter=None):
        self.root = Path(root_dir)
        self.filter = filter
        super().__init__("custom_ops", [])

    def __iter__(self):
        for optest in self.optests:
            if self.filter is not None:
                if (
                    optest.op.__name__ not in self.filter
                    or self.op_name_map[optest.op] not in self.filter
                ):
                    continue
            yield optest

    op_name_map = {}

    def add_test(self, op_name, impl_as_op, correctness_tests, performance_tests):
        self.op_name_map[impl_as_op] = op_name
        self.optests.append(OpTest(impl_as_op, correctness_tests, performance_tests))
