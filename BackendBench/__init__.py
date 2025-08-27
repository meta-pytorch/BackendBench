# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
BackendBench: A PyTorch backend evaluation framework.
"""

__version__ = "0.1.0"

# Export key utilities
from .op_mapper import PyTorchOpMapper, find_pytorch_ops, OperatorSchema

__all__ = ["PyTorchOpMapper", "find_pytorch_ops", "OperatorSchema"]
