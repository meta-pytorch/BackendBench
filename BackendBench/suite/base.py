# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


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
    def __init__(self, op, correctness_tests, performance_tests, ref_func=None):
        self.op = op
        self.correctness_tests = correctness_tests
        self.performance_tests = performance_tests
        self.ref_func = ref_func


class TestSuite:
    def __init__(self, name, optests):
        self.name = name
        self.optests = optests

    def __iter__(self):
        for optest in self.optests:
            yield optest
