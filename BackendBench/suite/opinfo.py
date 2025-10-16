# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict

from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._python_dispatch import TorchDispatchMode

from BackendBench.backwards_utils import (
    make_tensors_require_gradients,
    should_check_backwards_for_op,
)
from BackendBench.eval import allclose

from .base import OpTest, TestSuite

logger = logging.getLogger(__name__)


class OpInfoTest:
    def __init__(self, *args, test_backwards=False, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.test_backwards = test_backwards


class OpInfoOpTest(OpTest):
    def __init__(self, op, correctness_tests, indices, check_backwards=False):
        self.op = op
        self._correctness_tests = correctness_tests
        self.indices = set(indices)
        self.performance_tests = []
        self._check_backwards = check_backwards

    @property
    def correctness_tests(self):
        # Determine if this op should check backwards
        test_backwards = should_check_backwards_for_op(self.op.__name__, self._check_backwards)

        for idx, test in enumerate(self._correctness_tests):
            if idx in self.indices:
                # print(f"{idx} {test.input=} {test.args=} {test.kwargs=}")
                if test_backwards:
                    make_tensors_require_gradients(test.args, test.kwargs)
                yield OpInfoTest(
                    test.input, *test.args, test_backwards=test_backwards, **test.kwargs
                )


class OpTracerMode(TorchDispatchMode):
    def __init__(self):
        self.ops = []
        self.args = []
        self.kwargs = []

    def __torch_dispatch__(self, fn, types, args=(), kwargs={}):
        self.ops.append(fn)
        self.args.append(args)
        self.kwargs.append(kwargs)

        return fn(*args, **kwargs)


def build_op_tests(device, dtype, filter=None, check_backwards=False):
    op_info_op_tests = []
    for op in op_db:
        if filter and op.name not in filter:
            continue
        if "." in op.name and "nn.functional" not in op.name:
            continue
        if dtype not in op.supported_dtypes(device):
            continue
        if op.name in ["nonzero_static"]:
            continue

        op_indices = defaultdict(list)
        try:
            sample_inputs = list(op.sample_inputs(device, dtype))
        except Exception:
            continue

        for idx, test in enumerate(sample_inputs):
            # print(f"{idx=} {test.input=} {test.args=} {test.kwargs=}")
            try:
                with OpTracerMode() as tracer:
                    ref = op.op(test.input, *test.args, **test.kwargs)
                if len(tracer.ops) == 1:
                    res = tracer.ops[0](test.input, *test.args, **test.kwargs)
                    if allclose(ref, res):
                        op_indices[tracer.ops[0]].append(idx)
                else:
                    logger.debug(f"opinfo {op.name} has {len(tracer.ops)} ops")
            except Exception:
                continue

        for overload, indices in op_indices.items():
            if len(indices) > 0:
                op_info_op_tests.append(
                    OpInfoOpTest(overload, sample_inputs, indices, check_backwards)
                )

    return op_info_op_tests


class OpInfoTestSuite(TestSuite):
    def __init__(self, name, device, dtype, filter=None, check_backwards=False):
        super().__init__(name, build_op_tests(device, dtype, filter, check_backwards))
