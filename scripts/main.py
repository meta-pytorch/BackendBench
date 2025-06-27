import logging
import sys

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
def cli(suite, backend, ops):
    if ops:
        ops = ops.split(",")

    backend = {
        "aten": backends.AtenBackend,
        "flag_gems": backends.FlagGemsBackend,
        "llm": backends.LLMBackend,
    }[backend]()
    
    # For LLM backend, we need to generate kernels first
    if backend.name == "llm":
        llm_client = ClaudeKernelGenerator()
        backend = setup_llm_backend(backend, llm_client, suite, ops)

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


def setup_llm_backend(llm_backend, llm_client, suite_name, ops_filter):
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
            
        for op_test in suite:
            op = op_test.op
            op_name = str(op).split('.')[-1]
            op_signature = f"def {op_name}(*args, **kwargs) -> torch.Tensor"
            op_description = f"PyTorch operation: {op_name}"
            
            logger.info(f"Generating kernel for {op_name}")
            kernel_code = llm_client.generate_kernel(op_name, op_signature, op_description)
            
            try:
                llm_backend.add_kernel(op, kernel_code)
                logger.info(f"Successfully compiled kernel for {op_name}")
            except Exception as e:
                logger.error(f"Failed to compile kernel for {op_name}: {e}")
                # Continue with other operations
        
        return llm_backend
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)


if __name__ == "__main__":
    cli()
