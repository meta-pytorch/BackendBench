# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from BackendBench.opregistry import register_operator

from .base import Backend


class PrivateUse1Backend(Backend):
    def __init__(self) -> None:
        super().__init__("privateuse1")
        self.ops = {}

    def register_op(self, op_name: str, func) -> None:
        self.ops[op_name] = func
        register_operator(op_name)

    def is_registered(self, op_name: str) -> bool:
        return op_name in self.ops

    def __getitem__(self, op_name: str):
        if op_name in self.ops:
            return self.ops[op_name]
        raise KeyError(f"Operation {op_name} not found in {self.name} backend.")
