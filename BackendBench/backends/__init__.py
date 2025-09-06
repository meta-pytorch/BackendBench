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

import importlib.util

from .base_directory_backend import BaseDirectoryBackendABC
from .aten import AtenBackend
from .base import Backend
from .custom_ops import CustomOpsBackend
from .directory import DirectoryBackend
from .flag_gems import FlagGemsBackend
from .kernel_agent import KernelAgentBackend
from .llm import LLMBackend

__all__ = [
    "Backend",
    "BaseDirectoryBackendABC",
    "CustomOpsBackend",
    "DirectoryBackend",
    "AtenBackend",
    "FlagGemsBackend",
    "LLMBackend",
]

if importlib.util.find_spec("triton_kernel_agent") is not None:
    from .kernel_agent import KernelAgentBackend

    __all__.append("KernelAgentBackend")
else:
    KernelAgentBackend = None
