# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from BackendBench.opregistry import get_operator

from .base import OpTest, Test, TestSuite


def randn(*args, **kwargs):
    return lambda: torch.randn(*args, **kwargs)


SmokeTestSuite = TestSuite(
    "smoke",
    [
        OpTest(
            get_operator(torch.ops.aten.bmm.default),
            [
                Test(
                    randn(2, 2, 2, device="cpu"),
                    randn(2, 2, 2, device="cpu"),
                ),
            ],
            [
                Test(
                    randn(2**10, 2**10, 2**10, device="cpu"),
                    randn(2**10, 2**10, 2**10, device="cpu"),
                ),
            ],
        )
    ],
)
