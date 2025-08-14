import torch
import math
import pytest

from BackendBench.utils import (
    serialize_args,
    deserialize_args,
    _deserialize_tensor,
    uses_cuda_stream,
    check_for_constant_output,
    check_constant_inputs,
)

# Check if CUDA is available
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
        """Test detection of CuPy CUDA stream creation."""
        pytest.importorskip("cupy")
        import cupy

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

    def test_opoverload_callables(self):
        """Test that OpOverload objects don't raise exceptions."""
        import torch

        # Test OpOverload (torch operators)
        assert not uses_cuda_stream(torch.add)
        assert not uses_cuda_stream(torch.ops.aten.add)


class TestDeserializeArgs:
    """Test cases for deserialize_args function"""

    def test_single_tensor_arg(self):
        """Test deserializing a single tensor argument"""
        input_str = "((T([48, 24, 28, 28], f16),), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 1
        assert len(kwargs) == 0
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape == (48, 24, 28, 28)
        assert args[0].dtype == torch.float16
        # Device will be 'cuda' if available, otherwise 'cpu'
        assert args[0].device.type in ["cuda", "cpu"]

    def test_user_specified_input_1(self):
        """Test deserializing user-specified input case 1"""
        input_str = "((T([48, 24, 28, 28], f16),), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 1
        assert len(kwargs) == 0
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape == (48, 24, 28, 28)
        assert args[0].dtype == torch.float16
        assert args[0].device.type in ["cuda", "cpu"]

    def test_user_specified_input_2(self):
        """Test deserializing user-specified input case 2"""
        input_str = "((T([8, 8, 8, 8, 8], f16), T([8, 8, 8, 8, 8], f16),), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 2
        assert len(kwargs) == 0
        assert all(isinstance(arg, torch.Tensor) for arg in args)
        assert all(arg.shape == (8, 8, 8, 8, 8) for arg in args)
        assert all(arg.dtype == torch.float16 for arg in args)
        assert all(arg.device.type in ["cuda", "cpu"] for arg in args)

    def test_user_specified_input_3(self):
        """Test deserializing user-specified input case 3"""
        input_str = "((T([128, 256], f16), [1024, 249, 249],), {'dtype': torch.float16, 'layout': torch.strided, 'device': 'cuda'})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 2
        assert len(kwargs) == 3
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape == (128, 256)
        assert args[0].dtype == torch.float16
        assert args[1] == [1024, 249, 249]
        assert kwargs["dtype"] == torch.float16
        assert kwargs["layout"] == torch.strided
        assert kwargs["device"] == "cuda"

    def test_multiple_tensor_args(self):
        """Test deserializing multiple tensor arguments with smaller tensors"""
        input_str = "((T([5, 6, 7, 8, 9], f16), T([5, 6, 7, 8, 9], f16),), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 2
        assert len(kwargs) == 0
        assert all(isinstance(arg, torch.Tensor) for arg in args)
        assert all(arg.shape == (5, 6, 7, 8, 9) for arg in args)
        assert all(arg.dtype == torch.float16 for arg in args)
        assert all(arg.device.type in ["cuda", "cpu"] for arg in args)

    def test_tensor_with_negative_values(self):
        """Test deserializing with negative numbers in lists"""
        input_str = "((T([10, 20], f32), [-1, -2, -3],), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 2
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape == (10, 20)
        assert args[0].dtype == torch.float32
        assert args[1] == [-1, -2, -3]

    def test_different_dtypes(self):
        """Test deserializing tensors with different dtypes"""
        test_cases = [
            ("((T([10, 20], f32),), {})", torch.float32),
            ("((T([10, 20], f64),), {})", torch.float64),
            ("((T([10, 20], bf16),), {})", torch.bfloat16),
            ("((T([10, 20], i32),), {})", torch.int32),
            ("((T([10, 20], i64),), {})", torch.int64),
            ("((T([10, 20], b8),), {})", torch.bool),
        ]

        for input_str, expected_dtype in test_cases:
            args, kwargs = deserialize_args(input_str)
            assert args[0].dtype == expected_dtype

    def test_tensor_with_stride(self):
        """Test deserializing tensor with custom stride"""
        input_str = "((T([10, 20], f16, [40, 2]),), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 1
        tensor = args[0]
        assert tensor.shape == (10, 20)
        assert tensor.stride() == (40, 2)
        assert tensor.dtype == torch.float16

    def test_empty_args_kwargs(self):
        """Test deserializing empty args and kwargs"""
        input_str = "((), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 0
        assert len(kwargs) == 0

    def test_primitive_args(self):
        """Test deserializing primitive arguments"""
        input_str = "((1, 2.5, 'hello', True, None,), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 5
        assert args[0] == 1
        assert args[1] == 2.5
        assert args[2] == "hello"
        assert args[3] is True
        assert args[4] is None

    def test_math_inf(self):
        """Test deserializing math.inf"""
        input_str = "((inf,), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 1
        assert args[0] == math.inf

    def test_torch_constants(self):
        """Test deserializing torch constants"""
        input_str = "((torch.float16,), {})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 1
        assert args[0] == torch.float16

    def test_mixed_args_kwargs(self):
        """Test deserializing mixed args and kwargs"""
        input_str = "((T([5, 5], f32), 42,), {'alpha': 0.5, 'beta': T([3, 3], i64)})"
        args, kwargs = deserialize_args(input_str)

        assert len(args) == 2
        assert len(kwargs) == 2
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape == (5, 5)
        assert args[0].dtype == torch.float32
        assert args[1] == 42
        assert kwargs["alpha"] == 0.5
        assert isinstance(kwargs["beta"], torch.Tensor)
        assert kwargs["beta"].shape == (3, 3)
        assert kwargs["beta"].dtype == torch.int64


class TestSerializeArgs:
    """Test cases for serialize_args function"""

    def test_single_tensor_arg(self):
        """Test serializing a single tensor argument"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.randn(48, 24, 28, 28, dtype=torch.float16, device=device)
        args = (tensor,)
        kwargs = {}

        result = serialize_args(args, kwargs)
        expected = "((T([48, 24, 28, 28], f16),), {})"
        assert result == expected

    def test_multiple_tensor_args(self):
        """Test serializing multiple tensor arguments"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor1 = torch.randn(8, 8, 8, 8, 8, dtype=torch.float16, device=device)
        tensor2 = torch.randn(8, 8, 8, 8, 8, dtype=torch.float16, device=device)
        args = (tensor1, tensor2)
        kwargs = {}

        result = serialize_args(args, kwargs)
        expected = "((T([8, 8, 8, 8, 8], f16), T([8, 8, 8, 8, 8], f16),), {})"
        assert result == expected

    def test_tensor_with_list_and_kwargs(self):
        """Test serializing tensor with list and keyword arguments"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.randn(128, 256, dtype=torch.float16, device=device)
        args = (tensor, [1024, 249, 249])
        kwargs = {"dtype": torch.float16, "layout": torch.strided, "device": device}

        result = serialize_args(args, kwargs)
        expected = f"((T([128, 256], f16), [1024, 249, 249],), {{'dtype': torch.float16, 'layout': torch.strided, 'device': '{device}'}})"
        assert result == expected

    def test_different_dtypes(self):
        """Test reserializing tensors with different dtypes"""
        test_cases = [
            (torch.float32, "f32"),
            (torch.float64, "f64"),
            (torch.bfloat16, "bf16"),
            (torch.int32, "i32"),
            (torch.int64, "i64"),
            (torch.bool, "b8"),
        ]

        for dtype, expected_abbr in test_cases:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if dtype in [torch.int32, torch.int64, torch.bool]:
                tensor = torch.ones(10, 20, dtype=dtype, device=device)
            else:
                tensor = torch.randn(10, 20, dtype=dtype, device=device)
            args = (tensor,)
            kwargs = {}

            result = serialize_args(args, kwargs)
            expected = f"((T([10, 20], {expected_abbr}),), {{}})"
            assert result == expected

    def test_tensor_with_stride(self):
        """Test serializing tensor with custom stride"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.randn(20, 10, dtype=torch.float16, device=device)
        # Create a strided tensor
        strided_tensor = tensor.transpose(0, 1)  # This creates a non-contiguous tensor
        args = (strided_tensor,)
        kwargs = {}

        result = serialize_args(args, kwargs)
        # The exact stride depends on the tensor layout, but it should include stride info
        assert "T([10, 20], f16, [" in result
        assert "])" in result

    def test_empty_args_kwargs(self):
        """Test reserializing empty args and kwargs"""
        args = ()
        kwargs = {}

        result = serialize_args(args, kwargs)
        expected = "((), {})"
        assert result == expected

    def test_primitive_args(self):
        """Test reserializing primitive arguments"""
        args = (1, 2.5, "hello", True, None)
        kwargs = {}

        result = serialize_args(args, kwargs)
        expected = "((1, 2.5, 'hello', True, None,), {})"
        assert result == expected

    def test_none_inputs(self):
        """Test reserializing None inputs"""
        assert serialize_args(None, {}) == "None"
        assert serialize_args([], None) == "None"
        assert serialize_args(None, None) == "None"

    def test_list_with_tensors(self):
        """Test serializing list containing tensors"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor1 = torch.randn(5, 5, dtype=torch.float32, device=device)
        tensor2 = torch.ones(3, 3, dtype=torch.int64, device=device)  # Use ones for int tensor
        args = ([tensor1, tensor2, 42],)
        kwargs = {}

        result = serialize_args(args, kwargs)
        expected = "(([T([5, 5], f32), T([3, 3], i64), 42],), {})"
        assert result == expected

    def test_kwargs_with_tensors(self):
        """Test serializing kwargs containing tensors"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.randn(3, 3, dtype=torch.float32, device=device)
        args = ()
        kwargs = {"weight": tensor, "bias": None, "alpha": 0.5}

        result = serialize_args(args, kwargs)
        expected = "((), {'weight': T([3, 3], f32), 'bias': None, 'alpha': 0.5})"
        assert result == expected


class TestRoundTrip:
    """Test round-trip serialization/deserialization"""

    def test_roundtrip_single_tensor(self):
        """Test that serialize->deserialize produces equivalent tensors"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        original_tensor = torch.randn(10, 20, dtype=torch.float16, device=device)
        original_args = (original_tensor,)
        original_kwargs = {}

        # Serialize
        serialized = serialize_args(original_args, original_kwargs)

        # Deserialize
        deserialized_args, deserialized_kwargs = deserialize_args(serialized)

        # Check equivalence
        assert len(deserialized_args) == len(original_args)
        assert len(deserialized_kwargs) == len(original_kwargs)
        assert deserialized_args[0].shape == original_args[0].shape
        assert deserialized_args[0].dtype == original_args[0].dtype
        # Device type might differ due to CUDA availability fallback
        assert deserialized_args[0].device.type in ["cuda", "cpu"]

    def test_roundtrip_complex_args(self):
        """Test round-trip with complex arguments"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.randn(5, 5, dtype=torch.float32, device=device)
        original_args = (tensor, [1, 2, 3], "test")
        original_kwargs = {"alpha": 0.5, "beta": tensor}

        # Serialize
        serialized = serialize_args(original_args, original_kwargs)

        # Deserialize
        deserialized_args, deserialized_kwargs = deserialize_args(serialized)

        # Check equivalence
        assert len(deserialized_args) == len(original_args)
        assert len(deserialized_kwargs) == len(original_kwargs)
        assert deserialized_args[0].shape == original_args[0].shape
        assert deserialized_args[0].dtype == original_args[0].dtype
        assert deserialized_args[1] == original_args[1]
        assert deserialized_args[2] == original_args[2]
        assert deserialized_kwargs["alpha"] == original_kwargs["alpha"]
        assert deserialized_kwargs["beta"].shape == original_kwargs["beta"].shape
        assert deserialized_kwargs["beta"].dtype == original_kwargs["beta"].dtype


class TestDeserializeTensor:
    """Test cases for _deserialize_tensor helper function"""

    def test_basic_tensor_creation(self):
        """Test basic tensor creation with different dtypes"""
        tensor = _deserialize_tensor([10, 20], torch.float32)
        assert tensor.shape == (10, 20)
        assert tensor.dtype == torch.float32
        assert tensor.device.type in ["cuda", "cpu"]

    def test_tensor_with_stride(self):
        """Test tensor creation with custom stride"""
        tensor = _deserialize_tensor([5, 4], torch.float16, stride=[8, 2])
        assert tensor.shape == (5, 4)
        assert tensor.stride() == (8, 2)
        assert tensor.dtype == torch.float16

    def test_tensor_different_device(self):
        """Test tensor creation with different device"""
        tensor = _deserialize_tensor([3, 3], torch.float32, device="cpu")
        assert tensor.device.type == "cpu"

    def test_floating_point_range(self):
        """Test that floating point tensors have values in [0, 1] range"""
        for dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            tensor = _deserialize_tensor([100], dtype)
            assert tensor.min() >= 0
            assert tensor.max() <= 1

    def test_integer_tensors(self):
        """Test integer tensor creation"""
        for dtype in [torch.int32, torch.int64, torch.int8, torch.int16]:
            tensor = _deserialize_tensor([10], dtype)
            assert tensor.dtype == dtype
            assert tensor.shape == (10,)


class TestCheckForConstantOutput:
    """Test cases for check_for_constant_output function"""

    def test_constant_zeros_op(self):
        """Test that zeros creation is constant"""
        op = "aten.zeros"
        inps = "(([3, 4],), {'dtype': torch.float32})"

        result = check_for_constant_output(op, inps, n_iterations=5)
        assert result

    def test_unconstant_random_op(self):
        """Test that random operations are correctly detected as unconstant"""
        op = "aten.randn"
        inps = "(([3, 3],), {'dtype': torch.float32})"

        result = check_for_constant_output(op, inps, n_iterations=5)
        assert not result


class TestCheckConstantInputs:
    """Test cases for check_constant_inputs function"""

    def test_zeros_tensor(self):
        """Test detection of all-zeros and mostly-zeros tensors"""
        # Test all zeros
        zeros = torch.zeros(5, 5)
        args = (zeros,)
        kwargs = {}
        assert check_constant_inputs(args, kwargs)

        # Test mostly zeros (within threshold)
        mostly_zeros = torch.zeros(10, 10)
        mostly_zeros[0, 0] = 0.005  # Small deviation within default threshold of 0.01
        args = (mostly_zeros,)
        kwargs = {}
        assert check_constant_inputs(args, kwargs)

    def test_ones_tensor(self):
        """Test detection of all-ones and mostly-ones tensors"""
        # Test all ones
        ones = torch.ones(3, 4)
        args = (ones,)
        kwargs = {}
        assert check_constant_inputs(args, kwargs)

        # Test mostly ones (within threshold)
        mostly_ones = torch.ones(10, 10)
        mostly_ones[0, 0] = 0.995  # Small deviation within default threshold of 0.01
        args = (mostly_ones,)
        kwargs = {}
        assert check_constant_inputs(args, kwargs)

    def test_nan_tensor(self):
        """Test detection of tensor with NaN values"""
        nan_tensor = torch.tensor([1.0, float("nan"), 3.0])
        args = (nan_tensor,)
        kwargs = {}

        assert check_constant_inputs(args, kwargs)

    def test_random_tensor(self):
        """Test that random tensor is not flagged as constant"""
        random = torch.randn(5, 5)
        args = (random,)
        kwargs = {}

        assert not check_constant_inputs(args, kwargs)

    def test_kwargs_with_ones(self):
        """Test detection in kwargs"""
        ones = torch.ones(2, 2)
        args = ()
        kwargs = {"weight": ones}

        assert check_constant_inputs(args, kwargs)

    def test_mixed_inputs(self):
        """Test with both constant and non-constant tensors"""
        zeros = torch.zeros(3, 3)
        random = torch.randn(3, 3)
        args = (random, zeros)
        kwargs = {}

        # Should return True because at least one tensor is constant
        assert check_constant_inputs(args, kwargs)


if __name__ == "__main__":
    pytest.main([__file__])
