import pytest
import torch
import warnings

from BackendBench.eval import calculate_speed_of_light, get_gpu_specs


class TestSpeedOfLight:
    def test_gpu_specs_detection(self):
        """Test that GPU specs are detected correctly."""
        compute_peak, memory_bw = get_gpu_specs()
        assert compute_peak > 0
        assert memory_bw > 0

    def test_speed_of_light_realistic_performance(self):
        """Test that realistic performance doesn't trigger violations."""
        # Test with matrix multiply - realistic timing
        op = torch.ops.aten.mm.default
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        args = (a, b)
        kwargs = {}

        # 10ms is realistic for 100x100 matmul on CPU/GPU
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            efficiency = calculate_speed_of_light(op, args, kwargs, 10.0)

            # Should not trigger any warnings
            assert len(w) == 0

            # Should return reasonable efficiency (not None, not violation string)
            assert efficiency is not None
            assert isinstance(efficiency, float)
            assert 0 < efficiency < 1.0  # Should be reasonable percentage

    def test_speed_of_light_compute_violation(self):
        """Test that impossible compute performance triggers violation."""
        # Test with matrix multiply - impossibly fast timing
        op = torch.ops.aten.mm.default
        a = torch.randn(1000, 1000)  # Larger matrix for more FLOPs
        b = torch.randn(1000, 1000)
        args = (a, b)
        kwargs = {}

        # 0.001ms is impossibly fast for 1000x1000 matmul (2B FLOPs)
        efficiency = calculate_speed_of_light(op, args, kwargs, 0.001)

        # Should trigger compute violation
        assert isinstance(efficiency, str)
        assert "VIOLATION" in efficiency
        assert "compute" in efficiency

    def test_speed_of_light_memory_violation(self):
        """Test that impossible memory bandwidth triggers violation."""
        # Use ReLU which naturally has no FLOPs registered in PyTorch
        large_tensor = torch.randn(10_000_000)  # 40MB tensor
        args = (large_tensor,)
        kwargs = {}

        # 0.001ms is impossibly fast for moving 40MB of data
        efficiency = calculate_speed_of_light(torch.ops.aten.relu.default, args, kwargs, 0.001)

        # Should trigger memory violation
        assert isinstance(efficiency, str)
        assert "VIOLATION" in efficiency
        assert "memory" in efficiency

    def test_speed_of_light_no_flops_realistic(self):
        """Test memory-bound operation with realistic timing."""
        # Use ReLU which naturally has no FLOPs registered
        small_tensor = torch.randn(1000)  # 4KB tensor
        args = (small_tensor,)
        kwargs = {}

        # 1ms is reasonable for small memory operations
        efficiency = calculate_speed_of_light(torch.ops.aten.relu.default, args, kwargs, 1.0)

        # Should return reasonable memory efficiency
        assert isinstance(efficiency, float)
        assert 0 < efficiency < 1.0

    def test_speed_of_light_exception_handling(self):
        """Test that function handles exceptions gracefully."""
        # Invalid arguments that will cause an exception
        args = (5,)  # scalar argument to relu (which expects tensor)
        kwargs = {}

        efficiency = calculate_speed_of_light(torch.ops.aten.relu.default, args, kwargs, 1.0)

        # Should return None when operation fails
        assert efficiency is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests require CUDA")
class TestSpeedOfLightGPU:
    def test_t4_detection(self):
        """Test that T4 GPU is detected correctly in CI."""
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name.lower()

        compute_peak, memory_bw = get_gpu_specs()

        if "t4" in gpu_name:
            # Should detect T4 specs
            assert compute_peak == 65e12  # 65 TFLOPS
            assert memory_bw == 320e9  # 320 GB/s
        else:
            # Unknown GPU should use fallback
            assert compute_peak == 500e12
            assert memory_bw == 1000e9

    def test_gpu_realistic_matmul(self):
        """Test realistic GPU matrix multiply performance."""
        # Move tensors to GPU
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        args = (a, b)
        kwargs = {}

        # Warm up
        for _ in range(5):
            torch.mm(a, b)
        torch.cuda.synchronize()

        # 1ms should be reasonable for 512x512 matmul on GPU
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            efficiency = calculate_speed_of_light(torch.ops.aten.mm.default, args, kwargs, 1.0)

            # Should not trigger violations on real GPU timing
            assert len(w) == 0
            assert isinstance(efficiency, float)
            assert 0 < efficiency < 1.0
