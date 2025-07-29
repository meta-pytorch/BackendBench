from unittest.mock import Mock, patch

import pytest
import torch
from BackendBench.backends import (
    AtenBackend,
    FlagGemsBackend,
    KernelAgentBackend,
    LLMBackend,
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

        fake_op = Mock()
        fake_op.__module__ = "fake_module"
        assert fake_op in backend  # AtenBackend contains everything

    def test_aten_backend_getitem(self):
        backend = AtenBackend()

        relu_op = torch.ops.aten.relu.default
        assert backend[relu_op] == relu_op

        fake_op = Mock()
        fake_op.__module__ = "fake_module"
        assert backend[fake_op] == fake_op  # AtenBackend returns the op itself


class TestFlagGemsBackend:
    @pytest.mark.skipif(not HAS_FLAG_GEMS, reason="flag_gems not available")
    @patch("BackendBench.backends.flag_gems")
    def test_flag_gems_backend_initialization(self, mock_flag_gems):
        backend = FlagGemsBackend()
        assert backend.name == "flaggems"
        assert isinstance(backend.ops, dict)

    @pytest.mark.skipif(not HAS_FLAG_GEMS, reason="flag_gems not available")
    @patch("BackendBench.backends.flag_gems")
    def test_flag_gems_backend_contains_op(self, mock_flag_gems):
        mock_flag_gems.abs = Mock()

        backend = FlagGemsBackend()

        assert torch.ops.aten.abs.default in backend

        fake_op = Mock()
        fake_op.__str__ = Mock(return_value="fake_op")
        assert fake_op not in backend

    @pytest.mark.skipif(not HAS_FLAG_GEMS, reason="flag_gems not available")
    @patch("BackendBench.backends.flag_gems")
    def test_flag_gems_backend_getitem(self, mock_flag_gems):
        mock_abs_impl = Mock()
        mock_flag_gems.ops.abs = mock_abs_impl

        backend = FlagGemsBackend()

        assert backend[torch.ops.aten.abs.default] == mock_abs_impl

        fake_op = Mock()
        fake_op.__str__ = Mock(return_value="fake_op")
        with pytest.raises(KeyError):
            _ = backend[fake_op]


class TestLLMBackend:
    def test_llm_backend_initialization(self):
        with (
            patch("os.makedirs"),
            patch("builtins.open"),
            patch("datetime.datetime") as mock_datetime,
        ):
            mock_datetime.now.return_value.strftime.return_value = "20250721_204542"
            backend = LLMBackend()
            assert backend.name == "llm"
            assert "generated_kernels/run_" in backend.kernels_dir
            assert isinstance(backend.compiled_kernels, dict)

    @pytest.mark.skip(
        reason="Complex file I/O mocking needed - test requires full file system interaction"
    )
    def test_llm_backend_add_kernel(self):
        with (
            patch("os.makedirs"),
            patch("builtins.open"),
            patch("datetime.datetime") as mock_datetime,
        ):
            mock_datetime.now.return_value.strftime.return_value = "20250721_204542"
            backend = LLMBackend()

            mock_op = Mock()
            mock_op.__name__ = "test_op"

            kernel_code = """
def test_kernel(x):
    return x + 1
"""

            with patch("builtins.open", create=True) as mock_open:
                backend.add_kernel(mock_op, kernel_code, "test_op")

            mock_open.assert_called()

            assert mock_op in backend

    @pytest.mark.skip(
        reason="Complex file I/O mocking needed - test requires full file system interaction"
    )
    def test_llm_backend_test_kernel_correctness(self):
        with (
            patch("os.makedirs"),
            patch("builtins.open"),
            patch("datetime.datetime") as mock_datetime,
        ):
            mock_datetime.now.return_value.strftime.return_value = "20250721_204542"
            backend = LLMBackend()

            mock_op = Mock(return_value=torch.tensor([2.0]))

            kernel_code = """
def generated_kernel(x):
    return x + 1
"""

            mock_test = Mock()
            mock_test.args = [torch.tensor([1.0])]
            mock_test.kwargs = {}

            with patch("builtins.open", create=True):
                is_correct, feedback = backend.test_kernel_correctness(
                    mock_op, kernel_code, [mock_test], attempt=1
                )

            assert is_correct is True


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

    @pytest.mark.skipif(not HAS_KERNEL_AGENT, reason="KernelAgent not available")
    def test_kernel_agent_backend_generate_kernel(self):
        with (
            patch("triton_kernel_agent.TritonKernelAgent") as mock_kernel_agent_class,
        ):
            backend = KernelAgentBackend()

            mock_agent = Mock()
            mock_kernel_agent_class.return_value = mock_agent

            mock_agent.generate_kernel.return_value = {
                "success": True,
                "kernel_code": "def kernel(): pass",
                "rounds": 1,
                "session_dir": "test_session_dir",
                "worker_id": 0,
            }

            mock_op = Mock()
            mock_op.__str__ = Mock(return_value="test_op")
            with patch("builtins.open", create=True):
                kernel_code, success = backend.generate_kernel_with_agent(mock_op, "test_op")
            assert success is True
            assert kernel_code == "def kernel(): pass"
            mock_kernel_agent_class.assert_called_once()


class TestBackendIntegration:
    @pytest.mark.skipif(not HAS_FLAG_GEMS, reason="flag_gems not available")
    def test_backend_polymorphism(self):
        backends = []
        backends.append(AtenBackend())
        with patch("BackendBench.backends.flag_gems"):
            backends.append(FlagGemsBackend())
        backends.append(LLMBackend())
        backends.append(KernelAgentBackend())
        for backend in backends:
            assert hasattr(backend, "name")
            assert hasattr(backend, "__contains__")
            assert hasattr(backend, "__getitem__")
            assert isinstance(backend.name, str)
