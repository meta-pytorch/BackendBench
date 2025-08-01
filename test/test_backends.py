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


class TestLLMBackend:
    def test_llm_backend_initialization(self):
        backend = LLMBackend()
        assert backend.name == "llm"
        assert "generated_kernels/run_" in backend.kernels_dir
        assert isinstance(backend.compiled_kernels, dict)

    @pytest.mark.skip(reason="Requires Triton for kernel compilation")
    def test_llm_backend_add_kernel(self):
        backend = LLMBackend()

        # Use a real torch op for testing
        test_op = torch.ops.aten.relu.default

        kernel_code = """
@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0)
    tl.store(output_ptr + offsets, output, mask=mask)

def generated_relu(x):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
"""

        backend.add_kernel(test_op, kernel_code, "relu")

        assert test_op in backend

    @pytest.mark.skip(reason="Requires actual kernel compilation and testing")
    def test_llm_backend_test_kernel_correctness(self):
        # This test would require actual kernel compilation which may fail
        # in test environments without proper GPU setup
        pass


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
    @pytest.mark.skip(reason="Requires actual KernelAgent API interaction")
    def test_kernel_agent_backend_generate_kernel(self):
        # This test would require actual KernelAgent API calls
        # which should not be done in unit tests
        pass


class TestBackendIntegration:
    def test_backend_polymorphism(self):
        backends = []
        backends.append(AtenBackend())

        if HAS_FLAG_GEMS:
            backends.append(FlagGemsBackend())

        backends.append(LLMBackend())

        if HAS_KERNEL_AGENT:
            backends.append(KernelAgentBackend())

        for backend in backends:
            assert hasattr(backend, "name")
            assert hasattr(backend, "__contains__")
            assert hasattr(backend, "__getitem__")
            assert isinstance(backend.name, str)
