# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import os
import sys

import click
import torch

import BackendBench.backends as backends
import BackendBench.eval as eval
import BackendBench.multiprocessing_eval as multiprocessing_eval
from BackendBench.llm_client import LLMKernelGenerator, LLMRelayKernelGenerator
from BackendBench.output import save_results
from BackendBench.suite import (
    FactoTestSuite,
    ModelSuite,
    OpInfoTestSuite,
    SmokeTestSuite,
    TorchBenchTestSuite,
)

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
    type=click.Choice(["smoke", "opinfo", "torchbench", "facto", "model"]),
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
    help="Comma-separated list of ops to run (not supported for model suite)",
)
@click.option(
    "--model-filter",
    default=None,
    type=str,
    help="Comma-separated list of models to run (only for model suite)",
)
@click.option(
    "--topn-inputs",
    "--topn",
    default=None,
    type=int,
    help="Select the top N largest inputs for each op (default: all inputs)",
)
@click.option(
    "--llm-attempts",
    default=5,
    type=int,
    help="Attempts for LLM kernel generation with feedback",
)
@click.option(
    "--llm-model",
    default="claude-sonnet-4-20250514",
    type=str,
    help="Model name for LLM / LLM Relay backend (default: claude-sonnet-4-20250514). [Meta ONLY: `gcp-claude-4-sonnet` is a good choice for LLM Relay backend]",
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
    "--alternative-torchbench-data-path",
    default=None,
    type=str,
    help="Internal testing flag for BackendBench development. Users should not use this.",
)
@click.option(
    "--ops-directory",
    default="generated_kernels",
    type=str,
    help="Path to directory containing generated kernels",
)
@click.option(
    "--log-dir",
    default=None,
    type=str,
    help="Directory for output logs. Default: backendbench_output_{timestamp} (or ops-directory for directory backend)",
)
@click.option(
    "--disable-output-logs",
    default=False,
    is_flag=True,
    help="Disable verbose logging of individual test results",
)
@click.option(
    "--num-workers",
    default=None,
    type=int,
    help="Number of workers to use for multiprocessing, default to None to disable multiprocessing",
)
@click.option(
    "--check-overhead-dominated-ops",
    default=False,
    is_flag=True,
    help="Run tests for ops that are dominated by overhead ONLY",
)
@click.option(
    "--p",
    default=1.0,
    type=float,
    help=(
        "Performance score threshold for perf@p score calculation"
        "Note: Increasing this value makes the threshold more stringent, "
        "requiring a higher speedup to meet the performance criteria."
    ),
)
def cli(
    log_level,
    suite,
    backend,
    ops,
    model_filter,
    topn_inputs,
    llm_attempts,
    llm_model,
    kernel_agent_workers,
    kernel_agent_max_rounds,
    alternative_torchbench_data_path,
    ops_directory,
    log_dir,
    disable_output_logs,
    num_workers,
    check_overhead_dominated_ops,
    p,
):
    if suite != "torchbench":
        if topn_inputs is not None:
            raise ValueError("topn-inputs is only supported for torchbench suite")
        if check_overhead_dominated_ops:
            raise ValueError("check-overhead-dominated-ops is only supported for torchbench suite")

    if suite == "model":
        if backend != "directory":
            raise ValueError("model suite only supports directory backend")
        if ops is not None:
            raise ValueError(
                "--ops filter is not supported for model suite. Use --model-filter instead"
            )

    if suite != "model" and model_filter is not None:
        raise ValueError("--model-filter is only supported for model suite")

    setup_logging(log_level)
    if ops:
        ops = ops.split(",")
    if model_filter:
        model_filter = model_filter.split(",")

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
            alternative_torchbench_data_path,
            filter=ops,
            topn=topn_inputs,
            check_overhead_dominated_ops=check_overhead_dominated_ops,
        ),
        "facto": lambda: FactoTestSuite(
            "facto_cuda_bfloat16",
            "cuda",
            torch.bfloat16,
            filter=ops,
        ),
        "model": lambda: ModelSuite(filter=model_filter),
    }[suite]()

    backend_name = backend
    if backend == "llm-relay":
        llm_client = LLMRelayKernelGenerator(model=llm_model)
        backend = backends.LLMBackend(model=llm_model, llm_client=llm_client)
        backend.generate_kernels(suite, llm_attempts)
    elif backend == "llm":
        llm_client = LLMKernelGenerator(model=llm_model)
        backend = backends.LLMBackend(model=llm_model, llm_client=llm_client)
        backend.generate_kernels(suite, llm_attempts)
    elif backend == "kernel_agent":
        if backends.KernelAgentBackend is None:
            raise NotImplementedError("KernelAgent backend is for internal use only")
    elif backend == "directory":
        backend = backends.DirectoryBackend(ops_directory)
    else:
        backend = {
            "aten": backends.AtenBackend,
            "flag_gems": backends.FlagGemsBackend,
            "kernel_agent": backends.KernelAgentBackend,
            "directory": backends.DirectoryBackend,
        }[backend]()

    if not log_dir:
        if backend_name == "directory":
            # For directory backend, default to ops_directory
            log_dir = ops_directory
        else:
            # For other backends, create timestamped directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"backendbench_output_{timestamp}"

    overall_correctness = []
    overall_performance = []
    all_correctness_results = []
    all_performance_results = []

    if num_workers is None:
        for test in suite:
            if test.op not in backend:
                continue

            logger.debug(test.op)

            _, perf, correctness_results, performance_results = eval.eval_one_op(
                test.op,
                backend[test.op],
                test.correctness_tests,
                test.performance_tests,
            )

            overall_correctness.append(all(result.is_correct for result in correctness_results))
            overall_performance.append(perf)
            all_correctness_results.extend(correctness_results)
            all_performance_results.extend(performance_results)

            logger.debug(f"max memory allocated: {torch.cuda.max_memory_allocated():,}")
    else:
        with multiprocessing_eval.MultiprocessingEvaluator(num_workers) as evaluator:
            # Submit all tasks
            for test in suite:
                if test.op not in backend:
                    continue

                logger.debug(test.op)

                _ = evaluator.submit_task(
                    test.op,
                    backend[test.op],
                    test.correctness_tests,
                    test.performance_tests,
                )

            # Start evaluation
            evaluator.start_evaluation()

            # Get results
            results = evaluator.get_results()

        for result in results:
            correctness_score = all(
                correctness_result.is_correct for correctness_result in result.correctness_results
            )
            performance_score = result.performance_score
            overall_correctness.append(correctness_score)
            overall_performance.append(performance_score)
            all_correctness_results.extend(result.correctness_results)
            all_performance_results.extend(result.performance_results)

    # @todo: We should just calculate these in a seperate function from verbose_results
    mean_correctness = torch.tensor(overall_correctness).float().mean().item()
    geomean_perf = torch.tensor(overall_performance).log().mean().exp().item()
    perf_at_p_score = eval.perf_at_p(overall_correctness, overall_performance, p)

    print(f"correctness score (mean pass rate over all operators): {mean_correctness:.2f}")
    print(f"performance score (geomean speedup over all operators): {geomean_perf:.2f}")
    print(
        f"perf@p score (rate of correct samples with a speedup greater than p, p={p}): {perf_at_p_score:.2f}"
    )

    # Add full model testing for model suite
    if suite.name == "model":
        # Pass ops_directory as kernel_dir for directory backend
        kernel_dir = ops_directory if backend_name == "directory" else None
        model_results = suite.test_model_correctness(kernel_dir=kernel_dir)
        suite.print_model_correctness_results(model_results)

    command = "python -m BackendBench.scripts.main " + " ".join(sys.argv[1:])

    # Save results if not disabled

    if not disable_output_logs:
        save_results(
            all_correctness_results,
            all_performance_results,
            log_dir,
            command=command,
            mean_correctness=mean_correctness,
            geomean_perf=geomean_perf,
            perf_at_p_score=perf_at_p_score,
            p=p,
        )


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
        print(f"Successfully generated kernels for {successful_ops} ops")
        print(f"Failed to generate kernels for {total_ops - successful_ops} ops")
        print(
            f"Successful Kernel Generation rate: {successful_ops / total_ops * 100:.1f}%"
            if total_ops > 0
            else "Successful Kernel Generation rate: 0.0%"
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
        sys.exit(1)


if __name__ == "__main__":
    cli()
