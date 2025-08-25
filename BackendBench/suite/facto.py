# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

try:
    from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
    from facto.inputgen.utils.config import TensorConfig
    from facto.specdb.db import SpecDictDB
except ImportError:
    ArgumentTupleGenerator = None
    TensorConfig = None
    SpecDictDB = None


from BackendBench.eval import allclose
from BackendBench.opregistry import get_operator
from .base import OpTest, TestSuite

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


def build_facto_op_tests(device, dtype, filter=None, num_runs=10, empty=False, probability=1.0):
    facto_op_tests = []
    failed = []
    for spec_name in SpecDictDB:
        try:
            if filter and spec_name not in filter:
                continue

            # Get canonical operator from registry
            op = get_operator(spec_name)
            if op is None:
                logger.debug(f"Skipping {spec_name}: operator resolution failed")
                continue

            config = TensorConfig(
                empty=empty,
            ).set_probability(probability)

            spec = SpecDictDB[spec_name]
            generator = ArgumentTupleGenerator(spec, config)

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

                try:
                    # Trace execution to find underlying PyTorch ops
                    with OpTracerMode() as tracer:
                        ref = op(*filtered_posargs, **all_kwargs)
                except Exception:
                    logger.debug(f"FACTO spec {spec_name} couldn't run underlying op {op}")
                    continue

                # Check if we captured exactly one op (clean mapping)
                if len(tracer.ops) == 1:
                    try:
                        # Verify the traced op produces the same result
                        res = tracer.ops[0](*filtered_posargs, **all_kwargs)
                        if allclose(ref, res):
                            op_tests[tracer.ops[0]].append(
                                FactoTest(*filtered_posargs, **all_kwargs)
                            )
                    except Exception:
                        logger.debug(
                            f"FACTO spec {spec_name} couldn't run underlying op {tracer.ops[0]}"
                        )
                else:
                    logger.debug(f"FACTO spec {spec_name} has {len(tracer.ops)} ops")

            for traced_op, tests in op_tests.items():
                if len(tests) > 0:
                    facto_op_tests.append(FactoOpTest(traced_op, tests))
        except Exception:
            logger.debug(f"FACTO spec {spec_name} failed")
            failed.append(spec_name)

    logger.debug(f"Failed specs: {failed}")

    return facto_op_tests


class FactoTestSuite(TestSuite):
    def __init__(self, name, device, dtype, filter=None, num_runs=10, empty=False, probability=1.0):
        super().__init__(
            name, build_facto_op_tests(device, dtype, filter, num_runs, empty, probability)
        )
