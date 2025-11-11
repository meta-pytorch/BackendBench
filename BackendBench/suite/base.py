# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import importlib


class Test:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self):
        return [arg() for arg in self._args]

    @property
    def kwargs(self):
        return {k: v() for k, v in self._kwargs.items()}


class OpTest:
    def __init__(self, op, correctness_tests, performance_tests):
        self.op = op
        self.correctness_tests = correctness_tests
        self.performance_tests = performance_tests

    def __getstate__(self):
        # Custom serialization to handle callable op
        state = self.__dict__.copy()
        if callable(state.get("op")):
            op = state.pop("op")
            state["op_name"] = op.__name__
            state["op_module"] = op.__module__
        return state

    def __setstate__(self, state):
        if "op_name" in state and "op_module" in state:
            op_name = state.pop("op_name")
            op_module = state.pop("op_module")
            module = importlib.import_module(op_module)
            state["op"] = getattr(module, op_name)
        self.__dict__.update(state)


class TestSuite:
    def __init__(self, name, optests):
        self.name = name
        self.optests = optests

    def __iter__(self):
        for optest in self.optests:
            yield optest
