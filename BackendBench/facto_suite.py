import logging
from collections import defaultdict

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.specdb.db import SpecDictDB
from torch.utils._python_dispatch import TorchDispatchMode

from .eval import allclose
from .opregistry import get_operator
from .suite import OpTest, TestSuite

logger = logging.getLogger(__name__)


class FactoTest:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FactoOpTest(OpTest):
    def __init__(self, op, correctness_tests):
        self.op = op
        self._correctness_tests = correctness_tests
        self.performance_tests = []

    @property
    def correctness_tests(self):
        for test in self._correctness_tests:
            yield FactoTest(*test.args, **test.kwargs)


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


def build_facto_op_tests(device, dtype, filter=None, num_runs=10):
    facto_op_tests = []
    for spec_name in SpecDictDB:
        if filter and spec_name not in filter:
            continue

        # Get canonical operator from registry
        op = get_operator(spec_name)
        if op is None:
            logger.debug(f"Skipping {spec_name}: operator resolution failed")
            continue

        spec = SpecDictDB[spec_name]
        generator = ArgumentTupleGenerator(spec)

        op_tests = defaultdict(list)

        for idx, (posargs, inkwargs, outargs) in enumerate(generator.gen()):
            if idx >= num_runs:
                break

            # Filter arguments to target device/dtype
            filtered_posargs = []
            for arg in posargs:
                if isinstance(arg, torch.Tensor):
                    arg = arg.to(device=device, dtype=dtype)
                filtered_posargs.append(arg)

            filtered_inkwargs = {}
            for k, v in inkwargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device=device, dtype=dtype)
                filtered_inkwargs[k] = v

            filtered_outargs = {}
            for k, v in outargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device=device, dtype=dtype)
                filtered_outargs[k] = v

            all_kwargs = {**filtered_inkwargs, **filtered_outargs}

            # Trace execution to find underlying PyTorch ops
            with OpTracerMode() as tracer:
                ref = op(*filtered_posargs, **all_kwargs)

            # Check if we captured exactly one op (clean mapping)
            if len(tracer.ops) == 1:
                traced_op = tracer.ops[0]
                try:
                    # Verify the traced op produces the same result
                    res = traced_op(*filtered_posargs, **all_kwargs)
                    if allclose(ref, res):
                        op_tests[traced_op].append(
                            FactoTest(*filtered_posargs, **all_kwargs)
                        )
                except Exception:
                    logger.debug(
                        f"FACTO spec {spec_name} couldn't run underlying op {traced_op[0]}"
                    )
            else:
                logger.debug(f"FACTO spec {spec_name} has {len(tracer.ops)} ops")

        for traced_op, tests in op_tests.items():
            if len(tests) > 0:
                facto_op_tests.append(FactoOpTest(traced_op, tests))

    return facto_op_tests


class FactoTestSuite(TestSuite):
    def __init__(self, name, device, dtype, filter=None, num_runs=10):
        super().__init__(name, build_facto_op_tests(device, dtype, filter, num_runs))
