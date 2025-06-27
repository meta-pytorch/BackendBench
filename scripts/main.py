import logging
import sys

import BackendBench.backends as backends
import BackendBench.eval as eval
import click
import torch
from BackendBench.opinfo_suite import OpInfoTestSuite
from BackendBench.suite import SmokeTestSuite
from BackendBench.llm_client import ClaudeKernelGenerator
from BackendBench.llm_eval import full_eval

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
    "--llm-mode",
    default="generate",
    type=click.Choice(["generate", "evaluate"]),
    help="LLM mode: generate kernels and evaluate, or just evaluate existing kernels",
)
def cli(suite, backend, ops, llm_mode):
    if ops:
        ops = ops.split(",")

    # Handle LLM backend differently
    if backend == "llm":
        return run_llm_evaluation(suite, ops, llm_mode)
    
    backend = {
        "aten": backends.AtenBackend,
        "flag_gems": backends.FlagGemsBackend,
    }[backend]()

    suite = {
        "smoke": lambda: SmokeTestSuite,
        "opinfo": lambda: OpInfoTestSuite(
            "opinfo_cuda_bfloat16",
            "cuda",
            torch.bfloat16,
            filter=ops,
        ),
    }[suite]()

    overall_correctness = []
    overall_performance = []

    for test in suite:
        if test.op not in backend:
            continue

        logger.debug(test.op)

        correctness, perf = eval.eval_one_op(
            test.op,
            backend[test.op],
            test.correctness_tests,
            test.performance_tests,
        )
        overall_correctness.append(correctness)
        overall_performance.append(perf)

        logger.debug(f"max memory allocated: {torch.cuda.max_memory_allocated():,}")

    mean_correctness = torch.tensor(overall_correctness).mean().item()
    geomean_perf = torch.tensor(overall_performance).log().mean().exp().item()
    print(
        f"correctness score (mean pass rate over all operators): {mean_correctness:.2f}"
    )
    print(f"performance score (geomean speedup over all operators): {geomean_perf:.2f}")


def run_llm_evaluation(suite_name: str, ops_filter: list, llm_mode: str):
    """Run LLM-based kernel generation and evaluation."""
    
    # Initialize Claude client
    try:
        llm_client = ClaudeKernelGenerator()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Get list of operations to test
    if suite_name == "smoke":
        # For smoke test, just use relu
        test_ops = [torch.ops.aten.relu.default]
    elif suite_name == "opinfo":
        # Get ops from opinfo suite
        suite = OpInfoTestSuite("opinfo_cuda_bfloat16", "cuda", torch.bfloat16, filter=ops_filter)
        test_ops = [test.op for test in suite]
    else:
        print(f"Unknown suite: {suite_name}")
        return
    
    if ops_filter:
        # Filter operations based on name
        filtered_ops = []
        for op in test_ops:
            op_name = str(op).split('.')[-1]
            if op_name in ops_filter:
                filtered_ops.append(op)
        test_ops = filtered_ops
    
    print(f"Running LLM evaluation on {len(test_ops)} operations...")
    
    if llm_mode == "generate":
        # Generate and evaluate kernels
        score = full_eval(llm_client, test_ops, aggregation="geomean")
        print(f"LLM Backend Score (geomean speedup): {score:.2f}")
    else:
        print("Evaluate mode not yet implemented")


if __name__ == "__main__":
    cli()
