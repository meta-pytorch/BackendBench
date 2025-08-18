# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from .base import Backend


class AtenBackend(Backend):
    def __init__(self) -> None:
        super().__init__("aten")

    def __getitem__(self, key):
        return key

    def __contains__(self, key):
        return True
