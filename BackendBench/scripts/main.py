# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict

import BackendBench.backends as backends
import BackendBench.eval as eval
import BackendBench.multiprocessing_eval as multiprocessing_eval
import click
import torch

from BackendBench.facto_suite import FactoTestSuite
from BackendBench.llm_client import ClaudeKernelGenerator, LLMKernelGenerator
from BackendBench.opinfo_suite import OpInfoTestSuite
from BackendBench.suite import SmokeTestSuite
from BackendBench.torchbench_suite import DEFAULT_HUGGINGFACE_URL, TorchBenchTestSuite

logger = logging.getLogger(__name__)


def setup_logging(log_level):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s][%(levelname)s][%(filename)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.command()
@click.option(
    "--log-level",
    default=os.getenv("LOG_LEVEL", "INFO"),
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
@click.option(
    "--suite",
    default="smoke",
    type=click.Choice(["smoke", "opinfo", "torchbench", "facto"]),
    help="Which suite to run",
)
@click.option(
    "--backend",
    default="aten",
    type=click.Choice(["aten", "flag_gems", "llm", "llm-relay", "kernel_agent", "directory"]),
    help="Which backend to run",
)
@click.option(
    "--ops",
    default=None,
    type=str,
    help="Comma-separated list of ops to run",
)
@click.option(
    "--topn-inputs",
    "--topn",
    default=None,
    type=int,
    help="Select the top N largest inputs for each op (default: all inputs)",
)
@click.option(
    "--llm-max-attempts",
    default=5,
    type=int,
    help="Maximum attempts for LLM kernel generation with feedback",
)
@click.option(
    "--llm-relay-model",
    default="gcp-claude-4-sonnet",
    type=str,
    help="Model name for LLM Relay backend (default: gcp-claude-4-sonnet)",
)
@click.option(
    "--kernel-agent-workers",
    default=4,
    type=int,
    help="Number of parallel workers for KernelAgent backend",
)
@click.option(
    "--kernel-agent-max-rounds",
    default=10,
    type=int,
    help="Maximum refinement rounds per worker for KernelAgent backend",
)
@click.option(
    "--torchbench-data-path",
    default=DEFAULT_HUGGINGFACE_URL,
    type=str,
    help="Path to TorchBench operator data",
)
@click.option(
    "--ops-directory",
    default="generated_kernels",
    type=str,
    help="Path to directory containing generated kernels",
)
@click.option(
    "--output-path",
    default=None,
    type=str,
    help="Path for JSON output file with detailed results (if not specified, no JSON output)",
)
@click.option(
    "--num-workers",
    default=None,
    type=int,
    help="Number of workers to use for multiprocessing, default to None to disable multiprocessing",
)
def cli(
    log_level,
    suite,
    backend,
    ops,
    topn_inputs,
    llm_max_attempts,
    llm_relay_model,
    kernel_agent_workers,
    kernel_agent_max_rounds,
    torchbench_data_path,
    ops_directory,
    output_path,
    num_workers,
):
    setup_logging(log_level)
    if ops:
        ops = ops.split(",")

    if backend == "llm-relay":
        backend = backends.LLMRelayBackend(model=llm_relay_model)
    else:
        backend = {
            "aten": backends.AtenBackend,
            "flag_gems": backends.FlagGemsBackend,
            "llm": backends.LLMBackend,
            "kernel_agent": backends.KernelAgentBackend,
            "directory": backends.DirectoryBackend,
        }[backend]()

    suite = {
        "smoke": lambda: SmokeTestSuite,
        "opinfo": lambda: OpInfoTestSuite(
            "opinfo_cuda_bfloat16",
            "cuda",
            torch.bfloat16,
            filter=ops,
        ),
        "torchbench": lambda: TorchBenchTestSuite(
            "torchbench",
            torchbench_data_path,
            filter=ops,
            topn=topn_inputs,
        ),
        "facto": lambda: FactoTestSuite(
            "facto_cuda_bfloat16",
            "cuda",
            torch.bfloat16,
            filter=ops,
        ),
    }[suite]()

    # For LLM backend, we need to generate kernels first
    if backend.name == "llm":
        llm_client = ClaudeKernelGenerator()
        backend = setup_llm_backend(backend, llm_client, suite, llm_max_attempts)

    # For LLM Relay backend, we need to generate kernels using the local server
    elif backend.name == "llm-relay":
        llm_client = LLMKernelGenerator(model=llm_relay_model)
        backend = setup_llm_relay_backend(backend, llm_client, suite, llm_max_attempts)

    # For KernelAgent backend, we need to generate kernels using the sophisticated agent system
    elif backend.name == "kernel_agent":
        backend = setup_kernel_agent_backend(
            backend, suite, kernel_agent_workers, kernel_agent_max_rounds
        )

    # For Directory backend, we need to load existing kernels from a directory
    elif backend.name == "directory":
        backend = backends.DirectoryBackend(ops_directory)

    overall_correctness = []
    overall_performance = []
    verbose_results = []

    if num_workers is None:
        for test in suite:
            if test.op not in backend:
                continue

            logger.debug(test.op)

            correctness, perf, op_verbose_data = eval.eval_one_op(
                test.op,
                backend[test.op],
                test.correctness_tests,
                test.performance_tests,
            )
            overall_correctness.append(correctness)
            overall_performance.append(perf)

            # Convert dict to list entries with op_name
            op_name = getattr(test.op, "__name__", str(test.op))
            for args_str, data in op_verbose_data.items():
                entry = {"op_name": op_name, "args": args_str}
                entry.update(data)
                verbose_results.append(entry)      
      
            logger.debug(f"max memory allocated: {torch.cuda.max_memory_allocated():,}")
    else:
        with multiprocessing_eval.MultiprocessingEvaluator(num_workers) as evaluator:
            # Submit all tasks and track op names
            task_to_op_name = {}
            for test in suite:
                if test.op not in backend:
                    continue

                logger.debug(test.op)

                task_id = evaluator.submit_task(
                    test.op, backend[test.op], test.correctness_tests, test.performance_tests
                )
                op_name = getattr(test.op, "__name__", str(test.op))
                task_to_op_name[task_id] = op_name

            # Start evaluation
            evaluator.start_evaluation()

            # Get results
            results = evaluator.get_results()

        for result in results:
            correctness_score = result.correctness_score
            performance_score = result.performance_score
            overall_correctness.append(correctness_score)
            overall_performance.append(performance_score)
            
            # Handle verbose data if present
            if result.verbose_data and result.task_id in task_to_op_name:
                op_name = task_to_op_name[result.task_id]
                for args_str, data in result.verbose_data.items():
                    entry = {"op_name": op_name, "args": args_str}
                    entry.update(data)
                    verbose_results.append(entry)

    mean_correctness = torch.tensor(overall_correctness).mean().item()
    geomean_perf = torch.tensor(overall_performance).log().mean().exp().item()
    print(f"correctness score (mean pass rate over all operators): {mean_correctness:.2f}")
    print(f"performance score (geomean speedup over all operators): {geomean_perf:.2f}")

    # Save verbose results if output path is specified
    if output_path and verbose_results:
        eval.save_verbose_results(verbose_results, output_path)
        print(f"Detailed results saved to: {output_path}")


def setup_llm_backend(llm_backend, llm_client, suite, max_attempts=5):
    """Setup LLM backend by generating kernels for all operations in the suite."""
    try:
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

            # Generate kernel with iterative refinement
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


def setup_llm_relay_backend(llm_relay_backend, llm_client, suite, max_attempts=5):
    """Setup LLM Relay backend by generating kernels using the local plugboard server."""
    try:
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
                # TODO: Add performance testing in addition to correctness testing
                return llm_relay_backend.test_kernel_correctness(
                    op, kernel_code, op_test.correctness_tests, attempt
                )

            # Generate kernel with iterative refinement
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
                    llm_relay_backend.add_kernel(op, kernel_code, op_name)
                    print(
                        f"✓ Successfully generated and compiled kernel for {op_name} after {attempts_used} attempts"
                    )
                    successful_ops += 1

                    # Save summary of this operation
                    summary_file = os.path.join(
                        llm_relay_backend.kernels_dir, f"{op_name}_summary.txt"
                    )
                    with open(summary_file, "w") as f:
                        f.write(f"Operation: {op_name}\n")
                        f.write(f"Full op: {op_str}\n")
                        f.write(f"Attempts used: {attempts_used}/{max_attempts}\n")
                        f.write("Final status: Success\n")
                        f.write(f"Model: {llm_client.model}\n")
                        f.write(f"Server: {llm_client.server_url}\n")
                        f.write(f"Final kernel file: {op_name}_kernel_attempt_{attempts_used}.py\n")

                except Exception as e:
                    print(f"✗ Kernel passed tests but failed final compilation for {op_name}: {e}")
                    success = False

            if not success:
                print(f"✗ Skipping {op_name} - failed all {attempts_used} attempts")

                # Save summary of this operation
                summary_file = os.path.join(llm_relay_backend.kernels_dir, f"{op_name}_summary.txt")
                with open(summary_file, "w") as f:
                    f.write(f"Operation: {op_name}\n")
                    f.write(f"Full op: {op_str}\n")
                    f.write(f"Attempts used: {attempts_used}/{max_attempts}\n")
                    f.write("Final status: Failed - All attempts failed correctness tests\n")
                    f.write(f"Model: {llm_client.model}\n")
                    f.write(f"Server: {llm_client.server_url}\n")
                    f.write(f"Last kernel file: {op_name}_kernel_attempt_{attempts_used}.py\n")
                # Continue with other operations

        # Print summary
        print(f"\n{'=' * 60}")
        print("LLM RELAY BACKEND SETUP SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total operations: {total_ops}")
        print(f"Successful: {successful_ops}")
        print(f"Failed: {total_ops - successful_ops}")
        print(
            f"Success rate: {successful_ops / total_ops * 100:.1f}%"
            if total_ops > 0
            else "Success rate: 0.0%"
        )
        print(f"Model used: {llm_client.model}")
        print(f"Server: {llm_client.server_url}")
        print(f"Generated kernels saved to: {llm_relay_backend.kernels_dir}")
        print(f"{'=' * 60}\n")

        # Save overall summary
        overall_summary_file = os.path.join(llm_relay_backend.kernels_dir, "OVERALL_SUMMARY.txt")
        with open(overall_summary_file, "w") as f:
            f.write("LLM Relay Backend Generation Summary\n")
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
            f.write(f"Model: {llm_client.model}\n")
            f.write(f"Server: {llm_client.server_url}\n")
            f.write("Backend: LLM Relay (using local plugboard server)\n")

        return llm_relay_backend

    except Exception as e:
        print(f"Error setting up LLM Relay backend: {e}")
        if "Cannot connect to LLM relay server" in str(e):
            print("Please start the plugboard server with:")
            print(
                "buck run @//mode/inplace run_plugboard_server -- --model gcp-claude-4-sonnet --pipeline usecase-dev-ai-user"
            )
        sys.exit(1)


def setup_kernel_agent_backend(kernel_agent_backend, suite, num_workers=4, max_rounds=10):
    """Setup KernelAgent backend by generating kernels using the sophisticated agent system."""
    try:
        # Configure the backend with the specified parameters
        kernel_agent_backend.set_config(num_workers, max_rounds)

        successful_ops = 0
        total_ops = 0

        print(f"\n{'=' * 80}")
        print("KERNEL AGENT BACKEND SETUP")
        print(f"{'=' * 80}")
        print("Configuration:")
        print(f"  - Parallel workers: {num_workers}")
        print(f"  - Max refinement rounds per worker: {max_rounds}")
        print("  - Advanced features: Multi-turn dialogue, conversation history")
        print("  - Framework: OpenAI Triton with comprehensive guidelines")
        print(f"{'=' * 80}\n")

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

            print(f"\n[{total_ops}] {op_name.upper()} - KernelAgent Generation")
            print(f"    Operation: {op_str}")
            print(f"    Using {num_workers} parallel workers with up to {max_rounds} rounds each")

            # Generate kernel using KernelAgent's sophisticated system
            kernel_code, success = kernel_agent_backend.generate_kernel_with_agent(op, op_name)

            if success:
                try:
                    # Add the successful kernel to the backend
                    kernel_agent_backend.add_kernel(op, kernel_code, op_name)
                    print(f"✓ Successfully generated and compiled KernelAgent kernel for {op_name}")
                    successful_ops += 1

                    # Save summary of this operation
                    summary_file = os.path.join(
                        kernel_agent_backend.kernels_dir, f"{op_name}_summary.txt"
                    )
                    with open(summary_file, "w") as f:
                        f.write(f"Operation: {op_name}\n")
                        f.write(f"Full op: {op_str}\n")
                        f.write("Backend: KernelAgent\n")
                        f.write(f"Workers: {num_workers}\n")
                        f.write(f"Max rounds: {max_rounds}\n")
                        f.write("Final status: Success\n")
                        f.write("Generated using: Parallel workers + iterative refinement\n")

                except Exception as e:
                    print(
                        f"✗ KernelAgent generated kernel but compilation failed for {op_name}: {e}"
                    )
                    success = False

            if not success:
                print(f"✗ Skipping {op_name} - KernelAgent failed to generate working kernel")

                # Save summary of this operation
                summary_file = os.path.join(
                    kernel_agent_backend.kernels_dir, f"{op_name}_summary.txt"
                )
                with open(summary_file, "w") as f:
                    f.write(f"Operation: {op_name}\n")
                    f.write(f"Full op: {op_str}\n")
                    f.write("Backend: KernelAgent\n")
                    f.write(f"Workers: {num_workers}\n")
                    f.write(f"Max rounds: {max_rounds}\n")
                    f.write(
                        "Final status: Failed - KernelAgent could not generate working kernel\n"
                    )

        # Print summary
        print(f"\n{'=' * 80}")
        print("KERNEL AGENT BACKEND SETUP SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total operations: {total_ops}")
        print(f"Successful: {successful_ops}")
        print(f"Failed: {total_ops - successful_ops}")
        print(
            f"Success rate: {successful_ops / total_ops * 100:.1f}%"
            if total_ops > 0
            else "Success rate: 0.0%"
        )
        print(f"Generated kernels saved to: {kernel_agent_backend.kernels_dir}")
        print("Configuration used:")
        print(f"  - Parallel workers: {num_workers}")
        print(f"  - Max refinement rounds: {max_rounds}")
        print("  - Features: Triton guidelines, conversation history, auto test generation")
        print(f"{'=' * 80}\n")

        # Save overall summary
        overall_summary_file = os.path.join(kernel_agent_backend.kernels_dir, "OVERALL_SUMMARY.txt")
        with open(overall_summary_file, "w") as f:
            f.write("KernelAgent Backend Generation Summary\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Total operations: {total_ops}\n")
            f.write(f"Successful: {successful_ops}\n")
            f.write(f"Failed: {total_ops - successful_ops}\n")
            f.write(
                f"Success rate: {successful_ops / total_ops * 100:.1f}%\n"
                if total_ops > 0
                else "Success rate: 0.0%\n"
            )
            f.write(f"Parallel workers: {num_workers}\n")
            f.write(f"Max refinement rounds per worker: {max_rounds}\n")
            f.write("Advanced features used:\n")
            f.write("  - Multi-turn conversation with LLM\n")
            f.write("  - Comprehensive Triton programming guidelines\n")
            f.write("  - Automatic test generation and validation\n")
            f.write("  - Session management and artifact preservation\n")
            f.write("  - Parallel worker architecture for higher success rate\n")

        return kernel_agent_backend

    except Exception as e:
        print(f"Error setting up KernelAgent backend: {e}")
        if "OPENAI_API_KEY" in str(e) or "OpenAI" in str(e):
            print("Please set OPENAI_API_KEY environment variable")
        if "import" in str(e).lower():
            print("Please ensure KernelAgent is available in the parent directory")
        sys.exit(1)


if __name__ == "__main__":
    cli()
