# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from BackendBench.backends import (
    AtenBackend,
    FlagGemsBackend,
    KernelAgentBackend,
)

try:
    import importlib.util

    HAS_FLAG_GEMS = importlib.util.find_spec("flag_gems") is not None
except ImportError:
    HAS_FLAG_GEMS = False

try:
    import importlib.util
    import os
    import sys

    kernel_agent_path = os.path.join(os.path.dirname(__file__), "..", "KernelAgent")
    sys.path.insert(0, os.path.abspath(kernel_agent_path))
    HAS_KERNEL_AGENT = importlib.util.find_spec("triton_kernel_agent") is not None
except ImportError:
    HAS_KERNEL_AGENT = False


class TestAtenBackend:
    def test_aten_backend_initialization(self):
        backend = AtenBackend()
        assert backend.name == "aten"

    def test_aten_backend_contains_op(self):
        backend = AtenBackend()

        assert torch.ops.aten.relu.default in backend
        assert torch.ops.aten.add.Tensor in backend
        assert torch.ops.aten.mul.Tensor in backend

    def test_aten_backend_getitem(self):
        backend = AtenBackend()

        relu_op = torch.ops.aten.relu.default
        assert backend[relu_op] == relu_op

        add_op = torch.ops.aten.add.Tensor
        assert backend[add_op] == add_op


class TestFlagGemsBackend:
    @pytest.mark.skipif(not HAS_FLAG_GEMS, reason="flag_gems not available")
    def test_flag_gems_backend_initialization(self):
        backend = FlagGemsBackend()
        assert backend.name == "flaggems"
        assert isinstance(backend.ops, dict)

    @pytest.mark.skipif(not HAS_FLAG_GEMS, reason="flag_gems not available")
    def test_flag_gems_backend_contains_op(self):
        backend = FlagGemsBackend()

        # Test with actual ops that flag_gems supports
        if hasattr(torch.ops.aten, "abs"):
            if torch.ops.aten.abs.default in backend:
                assert torch.ops.aten.abs.default in backend

        # Test with an op that might not be in flag_gems
        unsupported_op = (
            torch.ops.aten.special_log_ndtr.default
            if hasattr(torch.ops.aten, "special_log_ndtr")
            else None
        )
        if unsupported_op:
            assert unsupported_op not in backend

    @pytest.mark.skipif(not HAS_FLAG_GEMS, reason="flag_gems not available")
    def test_flag_gems_backend_getitem(self):
        backend = FlagGemsBackend()

        # Test with an op that should exist
        if hasattr(torch.ops.aten, "abs") and torch.ops.aten.abs.default in backend:
            impl = backend[torch.ops.aten.abs.default]
            assert impl is not None

        # Test with an op that doesn't exist in flag_gems
        unsupported_op = (
            torch.ops.aten.special_log_ndtr.default
            if hasattr(torch.ops.aten, "special_log_ndtr")
            else None
        )
        if unsupported_op and unsupported_op not in backend:
            with pytest.raises(KeyError):
                _ = backend[unsupported_op]


class TestKernelAgentBackend:
    @pytest.mark.skipif(not HAS_KERNEL_AGENT, reason="KernelAgent not available")
    def test_kernel_agent_backend_initialization(self):
        backend = KernelAgentBackend()
        assert backend.name == "kernel_agent"
        assert "kernel_agent_run_" in backend.kernels_dir
        assert backend.num_workers == 4  # default value
        assert backend.max_rounds == 10  # default value

    @pytest.mark.skipif(not HAS_KERNEL_AGENT, reason="KernelAgent not available")
    def test_kernel_agent_backend_set_config(self):
        backend = KernelAgentBackend()

        backend.set_config(num_workers=8, max_rounds=20)

        assert backend.num_workers == 8
        assert backend.max_rounds == 20


class TestBackendIntegration:
    def test_backend_polymorphism(self):
        backends = []
        backends.append(AtenBackend())

        if HAS_FLAG_GEMS:
            backends.append(FlagGemsBackend())

        if HAS_KERNEL_AGENT:
            backends.append(KernelAgentBackend())

        for backend in backends:
            assert hasattr(backend, "name")
            assert hasattr(backend, "__contains__")
            assert hasattr(backend, "__getitem__")
            assert isinstance(backend.name, str)
