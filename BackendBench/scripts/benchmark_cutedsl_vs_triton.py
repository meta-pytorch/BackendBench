#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark script to compare CuteDSL vs Triton kernel implementations.
This script:
1. Runs create_cutedsl_ops.py to generate CuteDSL kernels
2. Renames the generated_kernels folder to generated_kernels_cutedsl
3. Runs create_triton_ops.py to generate Triton kernels
4. Renames the generated_kernels folder to generated_kernels_triton
5. Loads kernels from both folders and benchmarks performance using triton.testing.do_bench
6. Outputs a comparison table showing running times
"""

import importlib.util
import logging
import os
import shutil
import subprocess
import sys

import torch
import triton.testing
from tabulate import tabulate

logger = logging.getLogger(__name__)


def run_script(script_name):
    """Run a Python script and return the result."""
    try:
        subprocess.run([sys.executable, script_name], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run {script_name}: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def setup_kernel_directories():
    """Setup the kernel directories by running both scripts and organizing the outputs."""

    # Clean up any existing directories
    for dir_name in ["generated_kernels", "generated_kernels_cutedsl", "generated_kernels_triton"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            logger.info(f"Removed existing {dir_name} directory")

    # Step 1: Run CuteDSL script
    logger.info("Running create_cutedsl_ops.py...")
    if not run_script("BackendBench/scripts/create_cutedsl_ops.py"):
        raise RuntimeError("Failed to create CuteDSL kernels")

    # Step 2: Rename generated_kernels to generated_kernels_cutedsl
    if os.path.exists("generated_kernels"):
        shutil.move("generated_kernels", "generated_kernels_cutedsl")
    else:
        raise RuntimeError("CuteDSL script did not create generated_kernels directory")

    # Step 3: Run Triton script
    logger.info("Running create_triton_ops.py...")
    if not run_script("BackendBench/scripts/create_triton_ops.py"):
        raise RuntimeError("Failed to create Triton kernels")

    # Step 4: Rename generated_kernels to generated_kernels_triton
    if os.path.exists("generated_kernels"):
        shutil.move("generated_kernels", "generated_kernels_triton")
    else:
        raise RuntimeError("Triton script did not create generated_kernels directory")


def load_kernel_from_file(file_path, func_name):
    """Load a kernel function from a Python file."""
    try:
        spec = importlib.util.spec_from_file_location("kernel_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)
    except Exception as e:
        logger.error(f"Failed to load {func_name} from {file_path}: {e}")
        return None


def generate_test_inputs(op_name, shape=(1024, 1024), dtype=torch.float32):
    """Generate test inputs for the given operation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if op_name in ["relu", "abs"]:
        # Single input operations
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
        return [input_tensor]
    elif op_name in ["add", "mul"]:
        # Two input operations
        input1 = torch.randn(shape, dtype=dtype, device=device)
        input2 = torch.randn(shape, dtype=dtype, device=device)
        return [input1, input2]
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def benchmark_kernel(kernel_func, inputs, warmup=25, rep=100):
    """Benchmark a kernel function using triton.testing.do_bench."""
    if kernel_func is None:
        return float("inf")

    try:
        # Create a lambda that calls the kernel with the inputs
        kernel_call = lambda: kernel_func(*inputs)

        # Use triton.testing.do_bench for benchmarking
        ms = triton.testing.do_bench(kernel_call, warmup=warmup, rep=rep)
        return ms
    except Exception as e:
        logger.error(f"Failed to benchmark kernel: {e}")
        return float("inf")


def benchmark_precompiled_cutedsl_kernel(launch_func, inputs, warmup=25, rep=100):
    """Benchmark a precompiled CuteDSL kernel using triton.testing.do_bench."""
    if launch_func is None:
        return float("inf")

    try:
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        # Create output tensor
        if len(inputs) == 1:
            # Single input operations (relu, abs)
            output = torch.empty_like(inputs[0])
            a_ = from_dlpack(inputs[0])
            c_ = from_dlpack(output)

            # Precompile the kernel
            compiled_launch = cute.compile(launch_func, a_, c_)

            # Create benchmark function
            kernel_call = lambda: compiled_launch(a_, c_)

        elif len(inputs) == 2:
            # Two input operations (add, mul)
            output = torch.empty_like(inputs[0])
            a_ = from_dlpack(inputs[0])
            b_ = from_dlpack(inputs[1])
            c_ = from_dlpack(output)

            # Precompile the kernel
            compiled_launch = cute.compile(launch_func, a_, b_, c_)

            # Create benchmark function
            kernel_call = lambda: compiled_launch(a_, b_, c_)
        else:
            raise ValueError(f"Unsupported number of inputs: {len(inputs)}")

        # Use triton.testing.do_bench for benchmarking
        ms = triton.testing.do_bench(kernel_call, warmup=warmup, rep=rep)
        return ms
    except Exception as e:
        logger.error(f"Failed to benchmark precompiled CuteDSL kernel: {e}")
        return float("inf")


def get_cutedsl_launch_function_name(op_name):
    """Get the launch function name for a given operation in CuteDSL kernels."""
    launch_function_names = {
        "relu": "relu_kernel_launch",
        "add": "add_kernel_launch",
        "mul": "mul_kernel_launch",
        "abs": "abs_kernel_launch",
    }
    return launch_function_names.get(op_name)


def run_benchmarks():
    """Run benchmarks comparing CuteDSL vs Triton implementations."""

    operations = ["relu", "add", "mul", "abs"]
    results = []

    # Test with different tensor sizes
    test_shapes = [(512, 512), (1024, 1024), (2048, 2048)]

    for shape in test_shapes:
        shape_results = {"Shape": f"{shape[0]}x{shape[1]}", "Elements": shape[0] * shape[1]}

        for op in operations:
            # Load CuteDSL kernel wrapper function
            cutedsl_path = f"generated_kernels_cutedsl/{op}/{op}_implementation_v1.py"
            cutedsl_kernel = load_kernel_from_file(cutedsl_path, f"{op}_kernel_impl")

            # Load CuteDSL launch function for precompiled benchmarking
            launch_func_name = get_cutedsl_launch_function_name(op)
            cutedsl_launch_func = (
                load_kernel_from_file(cutedsl_path, launch_func_name) if launch_func_name else None
            )

            # Load Triton kernel
            triton_path = f"generated_kernels_triton/{op}/{op}_implementation_v1.py"
            triton_kernel = load_kernel_from_file(triton_path, f"{op}_kernel_impl")

            # Generate test inputs
            inputs = generate_test_inputs(op, shape)

            # Benchmark CuteDSL wrapper function
            cutedsl_time = benchmark_kernel(cutedsl_kernel, inputs)

            # Benchmark precompiled CuteDSL kernel
            cutedsl_precompiled_time = benchmark_precompiled_cutedsl_kernel(
                cutedsl_launch_func, inputs
            )

            # Benchmark Triton
            triton_time = benchmark_kernel(triton_kernel, inputs)

            # Store results
            shape_results[f"{op}_cutedsl_ms"] = f"{cutedsl_time:.3f}"
            shape_results[f"{op}_cutedsl_precompiled_ms"] = f"{cutedsl_precompiled_time:.3f}"
            shape_results[f"{op}_triton_ms"] = f"{triton_time:.3f}"

            # Calculate speedups (CuteDSL over Triton - higher is better for CuteDSL)
            if cutedsl_time != float("inf") and triton_time != float("inf") and cutedsl_time > 0:
                speedup = triton_time / cutedsl_time
                shape_results[f"{op}_speedup_cutedsl_vs_triton"] = f"{speedup:.2f}x"
            else:
                shape_results[f"{op}_speedup_cutedsl_vs_triton"] = "N/A"

            if (
                cutedsl_precompiled_time != float("inf")
                and triton_time != float("inf")
                and cutedsl_precompiled_time > 0
            ):
                speedup_precompiled = triton_time / cutedsl_precompiled_time
                shape_results[f"{op}_speedup_precompiled_vs_triton"] = f"{speedup_precompiled:.2f}x"
            else:
                shape_results[f"{op}_speedup_precompiled_vs_triton"] = "N/A"

        results.append(shape_results)

    return results


def print_results_table(results):
    """Print benchmark results in two separate formatted tables."""
    if not results:
        logger.error("No results to display")
        return

    operations = ["relu", "add", "mul", "abs"]

    # ===============================
    # Table 1: CuteDSL vs Triton
    # ===============================

    # Create headers for CuteDSL vs Triton table
    cutedsl_headers = ["Shape", "Elements"]
    for op in operations:
        cutedsl_headers.extend([f"{op}_cutedsl", f"{op}_triton", f"{op}_speedup"])

    # Prepare rows for CuteDSL vs Triton table
    cutedsl_table_rows = []
    for result in results:
        row = [result["Shape"], f"{result['Elements']:,}"]
        for op in operations:
            cutedsl_time = result.get(f"{op}_cutedsl_ms", "N/A")
            triton_time = result.get(f"{op}_triton_ms", "N/A")
            speedup_cutedsl_vs_triton = result.get(f"{op}_speedup_cutedsl_vs_triton", "N/A")

            row.extend([f"{cutedsl_time} ms", f"{triton_time} ms", speedup_cutedsl_vs_triton])
        cutedsl_table_rows.append(row)

    # Print CuteDSL vs Triton table
    print("\n" + "=" * 120)
    print("TABLE 1: CUTEDSL vs TRITON KERNEL BENCHMARK RESULTS")
    print("=" * 120)
    print(tabulate(cutedsl_table_rows, headers=cutedsl_headers, tablefmt="grid"))
    print("=" * 120)

    # ===============================
    # Table 2: Precompiled CuteDSL vs Triton
    # ===============================

    # Create headers for Precompiled vs Triton table
    precompiled_headers = ["Shape", "Elements"]
    for op in operations:
        precompiled_headers.extend([f"{op}_precompiled", f"{op}_triton", f"{op}_speedup"])

    # Prepare rows for Precompiled vs Triton table
    precompiled_table_rows = []
    for result in results:
        row = [result["Shape"], f"{result['Elements']:,}"]
        for op in operations:
            cutedsl_precompiled_time = result.get(f"{op}_cutedsl_precompiled_ms", "N/A")
            triton_time = result.get(f"{op}_triton_ms", "N/A")
            speedup_precompiled_vs_triton = result.get(f"{op}_speedup_precompiled_vs_triton", "N/A")

            row.extend(
                [
                    f"{cutedsl_precompiled_time} ms",
                    f"{triton_time} ms",
                    speedup_precompiled_vs_triton,
                ]
            )
        precompiled_table_rows.append(row)

    # Print Precompiled vs Triton table
    print("\n" + "=" * 120)
    print("TABLE 2: PRECOMPILED CUTEDSL vs TRITON KERNEL BENCHMARK RESULTS")
    print("=" * 120)
    print(tabulate(precompiled_table_rows, headers=precompiled_headers, tablefmt="grid"))
    print("=" * 120)

    # Print summary
    print("\nSUMMARY:")
    print("TABLE 1:")
    print("- CuteDSL times are wrapper function execution times (includes compilation overhead)")
    print("- Triton times are kernel execution times")
    print("- Speedup = Triton_time / CuteDSL_time (>1 means CuteDSL is faster)")
    print("\nTABLE 2:")
    print(
        "- Precompiled times are precompiled CuteDSL kernel execution times (no compilation overhead)"
    )
    print("- Triton times are kernel execution times")
    print("- Speedup = Triton_time / Precompiled_time (>1 means CuteDSL is faster)")
    print("\n- All times are in milliseconds")
    print("- N/A indicates benchmark failed for that kernel")


def main():
    """Main function to run the complete benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Starting CuteDSL vs Triton kernel benchmark...")

    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Benchmarks may not run properly.")

        # Setup kernel directories
        logger.info("Setting up kernel directories...")
        setup_kernel_directories()

        # Run benchmarks
        logger.info("Running benchmarks...")
        results = run_benchmarks()

        # Print results
        print_results_table(results)

        logger.info("Benchmark completed successfully!")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
