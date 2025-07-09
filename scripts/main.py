import logging
import os
import sys
from typing import Dict

import BackendBench.backends as backends
import BackendBench.eval as eval
import click
import torch
from BackendBench.opinfo_suite import OpInfoTestSuite
from BackendBench.suite import SmokeTestSuite
from BackendBench.llm_client import ClaudeKernelGenerator

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--suite",
    default="smoke",
    type=click.Choice(["smoke", "opinfo"]),
    help="Which suite to run",
)
@click.option(
    "--backend",
    default="aten",
    type=click.Choice(["aten", "flag_gems", "llm"]),
    help="Which backend to run",
)
@click.option(
    "--ops",
    default=None,
    type=str,
    help="Comma-separated list of ops to run",
)
@click.option(
    "--llm-max-attempts",
    default=5,
    type=int,
    help="Maximum attempts for LLM kernel generation with feedback",
)
@click.option(
    "--k",
    default=1,
    type=int,
    help="Number of trajectories to generate for sampling strategy (LLM backend only)",
)
def cli(suite, backend, ops, llm_max_attempts, k):
    if ops:
        ops = ops.split(",")

    # Validate that k is only used with LLM backend
    if k > 1 and backend != "llm":
        print(
            f"Error: --k parameter (k={k}) is only valid for LLM backend, not '{backend}' backend"
        )
        print("Use --backend llm to enable sampling strategy")
        sys.exit(1)

    backend = {
        "aten": backends.AtenBackend,
        "flag_gems": backends.FlagGemsBackend,
        "llm": backends.LLMBackend,
    }[backend]()

    # Generate k trajectories and report results for k=1, k=2, ..., k
    k_values = list(range(1, k + 1))

    # For LLM backend, we need to generate kernels first
    if backend.name == "llm":
        llm_client = ClaudeKernelGenerator()
        backend = setup_llm_backend(backend, llm_client, suite, ops, llm_max_attempts, k)

    suite = {
        "smoke": lambda: SmokeTestSuite,
        "opinfo": lambda: OpInfoTestSuite(
            "opinfo_cuda_bfloat16",
            "cuda",
            torch.bfloat16,
            filter=ops,
        ),
    }[suite]()

    # Track results per k value for sampling strategy analysis
    k_results = {}  # k -> {correctness_list, performance_list, success_count}
    if backend.name == "llm" and k > 1:
        k_values = list(range(1, k + 1))
        for k_val in k_values:
            k_results[k_val] = {"correctness": [], "performance": [], "success_count": 0}

    successful_correctness = []
    successful_performance = []
    total_ops = 0
    successful_ops = 0

    for test in suite:
        total_ops += 1

        if test.op not in backend:
            # Operation failed to generate kernel - count as failed
            continue

        logger.debug(test.op)
        successful_ops += 1

        correctness, perf = eval.eval_one_op(
            test.op,
            backend[test.op],
            test.correctness_tests,
            test.performance_tests,
        )
        successful_correctness.append(correctness)
        successful_performance.append(perf)

        # Track per-k results if using sampling strategy
        if backend.name == "llm" and k > 1 and hasattr(backend, "get_k_results"):
            op_k_results = backend.get_k_results(test.op)
            if op_k_results:
                for k_val in k_values:
                    if k_val in op_k_results:
                        k_results[k_val]["correctness"].append(op_k_results[k_val]["correctness"])
                        k_results[k_val]["performance"].append(op_k_results[k_val]["performance"])
                        k_results[k_val]["success_count"] += 1

        logger.debug(f"max memory allocated: {torch.cuda.max_memory_allocated():,}")

    # Calculate metrics
    generation_success_rate = successful_ops / total_ops if total_ops > 0 else 0.0

    if successful_ops > 0:
        mean_correctness_successful = torch.tensor(successful_correctness).mean().item()
        geomean_perf_successful = torch.tensor(successful_performance).log().mean().exp().item()
    else:
        mean_correctness_successful = 0.0
        geomean_perf_successful = 1.0

    # Overall metrics (including failed operations)
    mean_correctness_overall = mean_correctness_successful * generation_success_rate
    geomean_perf_overall = geomean_perf_successful * generation_success_rate + (
        1 - generation_success_rate
    )

    # Print comprehensive results
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Mean Correctness (successful): {mean_correctness_successful:.2f}")
    print(f"Geometric Mean Performance (successful): {geomean_perf_successful:.2f}x")
    print(f"Overall Correctness (including failures): {mean_correctness_overall:.2f}")
    print(f"Overall Performance (including failures): {geomean_perf_overall:.2f}x")

    # Print sampling strategy results if available
    if k_results:
        print(f"\n{'=' * 60}")
        print("SAMPLING STRATEGY RESULTS (per k value)")
        print(f"{'=' * 60}")
        for k_val in sorted(k_results.keys()):
            results = k_results[k_val]
            if results["correctness"]:
                mean_corr = torch.tensor(results["correctness"]).mean().item()
                geomean_perf = torch.tensor(results["performance"]).log().mean().exp().item()
                success_rate = results["success_count"] / total_ops
                print(
                    f"k={k_val:2d}: Mean Correctness={mean_corr:.3f}, Geomean Performance={geomean_perf:.3f}x, Success Rate={success_rate:.3f}"
                )
            else:
                print(f"k={k_val:2d}: No successful operations")

    print(f"{'=' * 60}")


def setup_llm_backend(llm_backend, llm_client, suite_name, ops_filter, max_attempts=5, k=1):
    """Setup LLM backend by generating kernels for all operations in the suite."""
    try:
        if suite_name == "smoke":
            suite = SmokeTestSuite
        elif suite_name == "opinfo":
            suite = OpInfoTestSuite(
                "opinfo_cuda_bfloat16",
                "cuda",
                torch.bfloat16,
                filter=ops_filter,
            )
        else:
            raise ValueError(f"Unknown suite: {suite_name}")

        successful_ops = 0
        total_ops = 0

        for op_test in suite:
            op = op_test.op
            total_ops += 1

            # Extract op name more carefully - e.g., torch.ops.aten.relu.default -> relu
            op_str = str(op)
            if "aten." in op_str:
                # Extract the operation name before any variant (like .default)
                op_name = op_str.split("aten.")[-1].split(".")[0]
            else:
                op_name = op_str.split(".")[-1]

            op_signature = f"def {op_name}(*args, **kwargs) -> torch.Tensor"
            op_description = f"PyTorch operation: {op_name}"

            print(
                f"\n[{total_ops}] Generating kernel for {op_name} (full op: {op_str}) with up to {max_attempts} attempts"
            )

            # Create feedback callback
            def feedback_callback(kernel_code: str, attempt: int) -> tuple[bool, Dict]:
                return llm_backend.test_kernel_correctness(
                    op, kernel_code, op_test.correctness_tests, attempt
                )

            # Generate kernels with sampling strategy if k > 1
            if k > 1:
                k_values = list(range(1, k + 1))
                print(
                    f"    Running sampling strategy: generating {k} trajectories, reporting for k={k_values}"
                )

                # Generate k trajectories
                all_trajectories = []
                for trajectory in range(k):
                    print(f"      Generating trajectory {trajectory + 1}/{k}...")

                    kernel_code, attempts_used, success = llm_client.generate_kernel_with_retry(
                        op_name,
                        op_signature,
                        op_description,
                        framework="triton",
                        max_attempts=max_attempts,
                        feedback_callback=feedback_callback,
                    )

                    if success:
                        # Test the kernel
                        correctness, feedback_info = llm_backend.test_kernel_correctness(
                            op, kernel_code, op_test.correctness_tests, 1
                        )
                        performance = feedback_info.get("performance_ratio", 1.0)

                        all_trajectories.append(
                            {
                                "kernel_code": kernel_code,
                                "correctness": correctness,
                                "performance": performance,
                                "attempts": attempts_used,
                            }
                        )

                # Report results for each k value and store for analysis
                best_kernel = None
                best_correctness = 0.0
                best_performance = 1.0
                op_k_results = {}

                for k_val in k_values:
                    if len(all_trajectories) >= k_val:
                        # Take first k_val trajectories
                        k_trajectories = all_trajectories[:k_val]
                        best_trajectory = max(k_trajectories, key=lambda t: t["correctness"])
                        print(
                            f"      k={k_val}: Best correctness={best_trajectory['correctness']:.2f}, performance={best_trajectory['performance']:.2f}x"
                        )

                        # Store results for this k value
                        op_k_results[k_val] = {
                            "correctness": best_trajectory["correctness"],
                            "performance": best_trajectory["performance"],
                        }

                        if best_trajectory["correctness"] > best_correctness:
                            best_kernel = best_trajectory["kernel_code"]
                            best_correctness = best_trajectory["correctness"]
                            best_performance = best_trajectory["performance"]
                    else:
                        print(
                            f"      k={k_val}: Not enough successful trajectories ({len(all_trajectories)} < {k_val})"
                        )

                # Store k results in backend for later analysis
                if op_k_results:
                    llm_backend.store_k_results(op, op_k_results)

                # Use the best kernel found across all k values
                if best_kernel:
                    kernel_code = best_kernel
                    success = True
                    attempts_used = max_attempts  # Approximate
                    print(
                        f"    Best overall: correctness={best_correctness:.2f}, performance={best_performance:.2f}x"
                    )
                else:
                    success = False
                    attempts_used = max_attempts
            else:
                # Original single trajectory approach
                kernel_code, attempts_used, success = llm_client.generate_kernel_with_retry(
                    op_name,
                    op_signature,
                    op_description,
                    framework="triton",
                    max_attempts=max_attempts,
                    feedback_callback=feedback_callback,
                )

            if success:
                try:
                    # Add the successful kernel to the backend
                    llm_backend.add_kernel(op, kernel_code, op_name)
                    print(
                        f"✓ Successfully generated and compiled kernel for {op_name} after {attempts_used} attempts"
                    )
                    successful_ops += 1

                    # Save summary of this operation
                    summary_file = os.path.join(llm_backend.kernels_dir, f"{op_name}_summary.txt")
                    with open(summary_file, "w") as f:
                        f.write(f"Operation: {op_name}\n")
                        f.write(f"Full op: {op_str}\n")
                        f.write(f"Attempts used: {attempts_used}/{max_attempts}\n")
                        f.write("Final status: Success\n")
                        f.write(f"Final kernel file: {op_name}_kernel_attempt_{attempts_used}.py\n")

                except Exception as e:
                    print(f"✗ Kernel passed tests but failed final compilation for {op_name}: {e}")
                    success = False

            if not success:
                print(f"✗ Skipping {op_name} - failed all {attempts_used} attempts")

                # Save summary of this operation
                summary_file = os.path.join(llm_backend.kernels_dir, f"{op_name}_summary.txt")
                with open(summary_file, "w") as f:
                    f.write(f"Operation: {op_name}\n")
                    f.write(f"Full op: {op_str}\n")
                    f.write(f"Attempts used: {attempts_used}/{max_attempts}\n")
                    f.write("Final status: Failed - All attempts failed correctness tests\n")
                    f.write(f"Last kernel file: {op_name}_kernel_attempt_{attempts_used}.py\n")
                # Continue with other operations

        # Print summary
        print(f"\n{'=' * 60}")
        print("LLM BACKEND SETUP SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total operations: {total_ops}")
        print(f"Successful: {successful_ops}")
        print(f"Failed: {total_ops - successful_ops}")
        print(
            f"Success rate: {successful_ops / total_ops * 100:.1f}%"
            if total_ops > 0
            else "Success rate: 0.0%"
        )
        print(f"Generated kernels saved to: {llm_backend.kernels_dir}")
        print(f"{'=' * 60}\n")

        # Print sampling strategy summary if used
        if k > 1:
            k_values = list(range(1, k + 1))
            print(f"\n{'=' * 60}")
            print("SAMPLING STRATEGY SUMMARY")
            print(f"{'=' * 60}")
            print(f"K values tested: {k_values}")
            print(f"Total operations: {total_ops}")
            print(f"Successful generations: {successful_ops}")
            print(f"Success rate: {successful_ops / total_ops * 100:.1f}%")
            print("Note: Results show best kernel found across all k values")
            print(f"{'=' * 60}\n")

        # Save overall summary
        overall_summary_file = os.path.join(llm_backend.kernels_dir, "OVERALL_SUMMARY.txt")
        with open(overall_summary_file, "w") as f:
            f.write("LLM Backend Generation Summary\n")
            f.write(f"{'=' * 40}\n")
            f.write(f"Total operations: {total_ops}\n")
            f.write(f"Successful: {successful_ops}\n")
            f.write(f"Failed: {total_ops - successful_ops}\n")
            f.write(
                f"Success rate: {successful_ops / total_ops * 100:.1f}%\n"
                if total_ops > 0
                else "Success rate: 0.0%\n"
            )
            f.write(f"Max attempts per operation: {max_attempts}\n")

        return llm_backend

    except Exception as e:
        print(f"Error setting up LLM backend: {e}")
        if "ANTHROPIC_API_KEY" in str(e):
            print("Please set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)


if __name__ == "__main__":
    cli()
