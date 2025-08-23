# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
BackendBench backends submodule.

This module provides various backend implementations for PyTorch operations.
Each backend implements a different strategy for mapping PyTorch operations
to alternative implementations.
"""

from .aten import AtenBackend
from .base import Backend
from .directory import DirectoryBackend
from .flag_gems import FlagGemsBackend
from .kernel_agent import KernelAgentBackend
from .kernel_agent_fp16 import KernelAgentFP16Backend
from .llm import LLMBackend
from .llm_relay import LLMRelayBackend

__all__ = [
    "Backend",
    "DirectoryBackend",
    "AtenBackend",
    "FlagGemsBackend",
    "LLMBackend",
    "LLMRelayBackend",
    "KernelAgentBackend",
    "KernelAgentFP16Backend",
]
