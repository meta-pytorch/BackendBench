import logging
from collections import defaultdict

import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._python_dispatch import TorchDispatchMode

from .eval import allclose
from .suite import OpTest, Test, TestSuite

logger = logging.getLogger(__name__)


class OpInfoTest:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class OpInfoOpTest(OpTest):
    def __init__(self, op, correctness_tests, indices):
        self.op = op
        self._correctness_tests = correctness_tests
        self.indices = set(indices)
        self.performance_tests = []

    @property
    def correctness_tests(self):
        for idx, test in enumerate(self._correctness_tests):
            if idx in self.indices:
                # print(f"{idx} {test.input=} {test.args=} {test.kwargs=}")
                yield OpInfoTest(test.input, *test.args, **test.kwargs)


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


def build_op_tests(device, dtype, filter=None):
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
        for idx, test in enumerate(op.sample_inputs(device, dtype)):
            # print(f"{idx=} {test.input=} {test.args=} {test.kwargs=}")
            with OpTracerMode() as tracer:
                ref = op.op(test.input, *test.args, **test.kwargs)
            if len(tracer.ops) == 1:
                try:
                    res = tracer.ops[0](test.input, *test.args, **test.kwargs)
                    if allclose(ref, res):
                        op_indices[tracer.ops[0]].append(idx)
                except Exception:
                    logger.debug(
                        f"opinfo {op.name} couldn't run underlying op {tracer.ops[0]}"
                    )
            else:
                logger.debug(f"opinfo {op.name} has {len(tracer.ops)} ops")

        for overload, indices in op_indices.items():
            if len(indices) > 0:
                op_info_op_tests.append(
                    OpInfoOpTest(overload, op.sample_inputs(device, dtype), indices)
                )

    return op_info_op_tests


class OpInfoTestSuite(TestSuite):
    def __init__(self, name, device, dtype, filter=None):
        super().__init__(name, build_op_tests(device, dtype, filter))
