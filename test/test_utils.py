import pytest
from BackendBench.utils import uses_cuda_stream

# Check if CUDA is available
import torch

HAS_CUDA = torch.cuda.is_available()


class TestCudaStreamDetection:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_pytorch_stream_creation(self):
        """Test detection of PyTorch CUDA stream creation."""

        def func_with_pytorch_stream():
            import torch

            stream = torch.cuda.Stream()
            return stream

        assert uses_cuda_stream(func_with_pytorch_stream)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_cupy_stream_creation(self):
        import cupy

        """Test detection of CuPy CUDA stream creation."""

        def func_with_cupy_stream():
            stream = cupy.cuda.Stream()
            return stream

        assert uses_cuda_stream(func_with_cupy_stream)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_generic_stream_creation(self):
        """Test detection of generic Stream() calls."""

        def func_with_generic_stream():
            from torch.cuda import Stream

            stream = Stream()
            return stream

        assert uses_cuda_stream(func_with_generic_stream)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_stream_with_device_id(self):
        """Test detection of Stream with device ID."""

        def func_with_device_stream():
            from torch.cuda import Stream

            stream = Stream(0)
            return stream

        assert uses_cuda_stream(func_with_device_stream)

    def test_no_stream_creation(self):
        """Test functions without stream creation return False."""

        def func_without_stream():
            import torch

            x = torch.randn(100, 100)
            y = x @ x.T
            return y

        assert not uses_cuda_stream(func_without_stream)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_lambda_function(self):
        """Test detection in lambda functions."""

        def func_lambda_with_stream():
            return torch.cuda.Stream()

        def func_lambda_without(x):
            return x * 2

        assert uses_cuda_stream(func_lambda_with_stream)
        assert not uses_cuda_stream(func_lambda_without)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_nested_function(self):
        """Test detection in nested functions."""

        def outer_function():
            def inner_with_stream():
                import torch

                return torch.cuda.Stream()

            return inner_with_stream

        inner = outer_function()
        assert uses_cuda_stream(inner)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_class_method(self):
        """Test detection in class methods."""

        class StreamClass:
            def method_with_stream(self):
                import torch

                self.stream = torch.cuda.Stream()
                return self.stream

            def method_without_stream(self):
                return "no stream here"

        obj = StreamClass()
        assert uses_cuda_stream(obj.method_with_stream)
        assert not uses_cuda_stream(obj.method_without_stream)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_various_formats(self):
        """Test various formatting of stream creation."""

        def func_spaces():
            stream = torch.cuda.Stream()
            return stream

        def func_multiline():
            stream = torch.cuda.Stream(device=0)
            return stream

        def func_chained():
            result = torch.cuda.Stream().query()
            return result

        assert uses_cuda_stream(func_spaces)
        assert uses_cuda_stream(func_multiline)
        assert uses_cuda_stream(func_chained)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_case_sensitivity(self):
        """Test case-insensitive detection."""

        def func_lowercase():
            stream = torch.cuda.stream()  # lowercase (if it existed)
            return stream

        def func_uppercase():
            stream = torch.cuda.STREAM()  # uppercase (if it existed)
            return stream

        # These should still be detected due to case-insensitive regex
        assert uses_cuda_stream(func_lowercase)
        assert uses_cuda_stream(func_uppercase)


if __name__ == "__main__":
    pytest.main([__file__])
