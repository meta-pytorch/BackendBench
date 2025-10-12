#!/usr/bin/env python3

"""Debug script to reproduce the exact eval_performance issue"""

import torch
import time
from BackendBench.backends.aten import AtenBackend
from BackendBench.eval import cpu_bench, allclose, compute_errors
from BackendBench.suite.smoke import SmokeTestSuite
from BackendBench.utils import serialize_args

def debug_eval_performance_issue():
    """Reproduce the exact flow of eval_performance to identify the issue"""

    print("=== Debugging eval_performance Issue ===")

    # Get the setup
    backend = AtenBackend()
    relu_test = next(test for test in SmokeTestSuite if "relu" in str(test.op))

    op = relu_test.op  # Reference operation
    impl = backend[relu_test.op]  # Implementation (should be same)
    test = relu_test.performance_tests[0]  # The performance test

    print(f"Operation: {op}")
    print(f"Implementation: {impl}")
    print(f"Same object: {op is impl}")
    print()

    # Cache the arguments to ensure consistency (from eval_performance line 369)
    cached_args = test.args
    cached_kwargs = test.kwargs
    args_str = serialize_args(cached_args, cached_kwargs)

    print(f"Cached args: {type(cached_args)}, length: {len(cached_args)}")
    print(f"Args str: {args_str}")

    # Create actual tensor (since args might be callable)
    actual_args = []
    for arg in cached_args:
        if callable(arg):
            actual_arg = arg()
            print(f"Callable arg generated tensor: {actual_arg.shape}, {actual_arg.device}")
            actual_args.append(actual_arg)
        else:
            actual_args.append(arg)

    cached_args = tuple(actual_args)
    print()

    # Step 1: Measure base_time (reference) - line 374
    print("=== Step 1: Measuring base_time (reference) ===")
    print("About to call: bench_fn(lambda: op(*cached_args, **cached_kwargs))")

    base_time = cpu_bench(lambda: op(*cached_args, **cached_kwargs))
    print(f"base_time: {base_time*1000:.6f}ms")
    print()

    # Step 2: Initial test_time assignment - line 378
    test_time = base_time
    print(f"Initial test_time (set to base_time): {test_time*1000:.6f}ms")
    print()

    # Step 3: Correctness check - lines 380-389
    print("=== Step 3: Correctness check ===")
    print("Running: ref = op(*cached_args, **cached_kwargs)")
    ref = op(*cached_args, **cached_kwargs)
    print(f"Reference result shape: {ref.shape}")

    print("Running: res = impl(*cached_args, **cached_kwargs)")
    res = impl(*cached_args, **cached_kwargs)
    print(f"Implementation result shape: {res.shape}")

    # Check if results are close
    is_close = allclose(ref, res)
    print(f"Results are close: {is_close}")

    if not is_close:
        abs_error, rel_error = compute_errors(ref, res)
        print(f"ERROR: Results not close! abs_error: {abs_error}, rel_error: {rel_error}")
        return
    print()

    # Step 4: Measure implementation time - line 390
    print("=== Step 4: Measuring implementation time ===")
    print("About to call: bench_fn(lambda: impl(*cached_args, **cached_kwargs))")

    measured_test_time = cpu_bench(lambda: impl(*cached_args, **cached_kwargs))
    print(f"measured test_time: {measured_test_time*1000:.6f}ms")

    # Update test_time
    test_time = measured_test_time
    print(f"Final test_time: {test_time*1000:.6f}ms")
    print()

    # Calculate speedup
    speedup = base_time / test_time
    print(f"=== Results ===")
    print(f"base_time: {base_time*1000:.6f}ms")
    print(f"test_time: {test_time*1000:.6f}ms")
    print(f"speedup: {speedup:.6f}")
    print()

    # Let's also check if there are any side effects by running multiple times
    print("=== Multiple Measurements ===")
    for i in range(5):
        bt = cpu_bench(lambda: op(*cached_args, **cached_kwargs))
        tt = cpu_bench(lambda: impl(*cached_args, **cached_kwargs))
        print(f"Run {i+1}: base={bt*1000:.6f}ms, test={tt*1000:.6f}ms, speedup={bt/tt:.6f}")

if __name__ == "__main__":
    debug_eval_performance_issue()