# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


class AgentError(Exception):
    """
    Exception raised for errors related to LLM/agent failures,
    such as rate limits, empty code, bad formatting, or API issues.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
