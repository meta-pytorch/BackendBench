#!/usr/bin/env python3

"""Debug script to investigate ReLU performance issue"""

import torch
import time
import gc
from BackendBench.backends.aten import AtenBackend
from BackendBench.eval import eval_one_op, cpu_bench
from BackendBench.suite.smoke import SmokeTestSuite

def debug_relu_performance():
    print("=== Debugging ReLU Performance Issue ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()

    # Get the AtenBackend
    backend = AtenBackend()

    # Find the ReLU test in SmokeTestSuite
    relu_test = None
    for test in SmokeTestSuite:
        if "relu" in str(test.op):
            relu_test = test
            break

    if not relu_test:
        print("ERROR: Could not find ReLU test in SmokeTestSuite")
        return

    print(f"Found test: {relu_test.op}")
    print(f"Correctness tests: {len(relu_test.correctness_tests)}")
    print(f"Performance tests: {len(relu_test.performance_tests)}")
    print()

    # Get the actual operations
    reference_op = relu_test.op
    impl_op = backend[relu_test.op]

    print(f"Reference op: {reference_op}")
    print(f"Implementation op: {impl_op}")
    print(f"Are they the same object? {reference_op is impl_op}")
    print()

    # Test the performance test case specifically
    perf_test = relu_test.performance_tests[0]

    print("=== Performance Test Details ===")
    print(f"Performance test args: {perf_test.args}")
    print(f"Performance test kwargs: {perf_test.kwargs}")

    # Create test tensor
    test_tensor = perf_test.args[0]() if callable(perf_test.args[0]) else perf_test.args[0]
    print(f"Test tensor shape: {test_tensor.shape}")
    print(f"Test tensor device: {test_tensor.device}")
    print(f"Test tensor dtype: {test_tensor.dtype}")
    print()

    # Manual timing comparison
    print("=== Manual Timing ===")

    # Warmup
    for _ in range(10):
        _ = reference_op(test_tensor)
        _ = impl_op(test_tensor)

    # Time reference
    start_time = time.perf_counter()
    for _ in range(100):
        result_ref = reference_op(test_tensor)
    ref_time = (time.perf_counter() - start_time) / 100

    # Time implementation
    start_time = time.perf_counter()
    for _ in range(100):
        result_impl = impl_op(test_tensor)
    impl_time = (time.perf_counter() - start_time) / 100

    print(f"Reference time per call: {ref_time*1000:.4f}ms")
    print(f"Implementation time per call: {impl_time*1000:.4f}ms")
    print(f"Manual speedup: {ref_time/impl_time:.4f}")
    print()

    # Check results are the same
    print(f"Results are close: {torch.allclose(result_ref, result_impl)}")
    print()

    # Try cpu_bench function
    print("=== Using cpu_bench function ===")
    ref_bench_time = cpu_bench(lambda: reference_op(test_tensor))
    impl_bench_time = cpu_bench(lambda: impl_op(test_tensor))

    print(f"cpu_bench reference time: {ref_bench_time*1000:.4f}ms")
    print(f"cpu_bench implementation time: {impl_bench_time*1000:.4f}ms")
    print(f"cpu_bench speedup: {ref_bench_time/impl_bench_time:.4f}")
    print()

    # Run the full eval_one_op to see what happens
    print("=== Full eval_one_op ===")
    correctness, perf, correctness_results, performance_results = eval_one_op(
        reference_op,
        impl_op,
        relu_test.correctness_tests,
        relu_test.performance_tests,
    )

    print(f"Correctness score: {correctness}")
    print(f"Performance score: {perf}")
    print()

    # Print performance results in detail
    for i, result in enumerate(performance_results):
        print(f"Performance test {i}:")
        print(f"  Speedup: {result.speedup}")
        print(f"  Reference time: {result.reference_time_ms}ms")
        print(f"  Benchmark time: {result.benchmark_time_ms}ms")
        print(f"  Successfully ran: {result.successfully_ran}")
        if result.error_msg:
            print(f"  Error: {result.error_msg}")
        print()

if __name__ == "__main__":
    debug_relu_performance()