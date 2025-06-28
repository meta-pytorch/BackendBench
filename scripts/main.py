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
def cli(suite, backend, ops, llm_max_attempts):
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
        backend = setup_llm_backend(backend, llm_client, suite, ops, llm_max_attempts)

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
    print(f"correctness score (mean pass rate over all operators): {mean_correctness:.2f}")
    print(f"performance score (geomean speedup over all operators): {geomean_perf:.2f}")


def setup_llm_backend(llm_backend, llm_client, suite_name, ops_filter, max_attempts=5):
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
            # Extract op name more carefully - e.g., torch.ops.aten.relu.default -> relu
            op_str = str(op)
            if 'aten.' in op_str:
                # Extract the operation name before any variant (like .default)
                op_name = op_str.split('aten.')[-1].split('.')[0]
            else:
                op_name = op_str.split('.')[-1]
            
            op_signature = f"def {op_name}(*args, **kwargs) -> torch.Tensor"
            op_description = f"PyTorch operation: {op_name}"
            
            print(f"Generating kernel for {op_name} (full op: {op_str}) with up to {max_attempts} attempts")
            
            # Create feedback callback
            def feedback_callback(kernel_code: str, attempt: int) -> tuple[bool, Dict]:
                return llm_backend.test_kernel_correctness(op, kernel_code, op_test.correctness_tests, attempt)
            
            # Generate kernel with iterative refinement
            kernel_code, attempts_used = llm_client.generate_kernel_with_retry(
                op_name, op_signature, op_description,
                framework="triton",
                max_attempts=max_attempts,
                feedback_callback=feedback_callback
            )
            
            try:
                # Add the final successful kernel to the backend
                llm_backend.add_kernel(op, kernel_code, op_name)
                print(f"✓ Successfully compiled kernel for {op_name} after {attempts_used} attempts")
                
                # Save summary of this operation
                summary_file = os.path.join(llm_backend.kernels_dir, f"{op_name}_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(f"Operation: {op_name}\n")
                    f.write(f"Full op: {op_str}\n")
                    f.write(f"Attempts used: {attempts_used}/{max_attempts}\n")
                    f.write(f"Final status: Success\n")
                    f.write(f"Final kernel file: {op_name}_kernel_attempt_{attempts_used}.py\n")
                    
            except Exception as e:
                print(f"✗ Failed to compile final kernel for {op_name}: {e}")
                
                # Save summary of this operation
                summary_file = os.path.join(llm_backend.kernels_dir, f"{op_name}_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(f"Operation: {op_name}\n")
                    f.write(f"Full op: {op_str}\n")
                    f.write(f"Attempts used: {attempts_used}/{max_attempts}\n")
                    f.write(f"Final status: Failed - {str(e)}\n")
                    f.write(f"Last kernel file: {op_name}_kernel_attempt_{attempts_used}.py\n")
                # Continue with other operations
        
        return llm_backend
        
    except Exception as e:
        print(f"Error setting up LLM backend: {e}")
        if "ANTHROPIC_API_KEY" in str(e):
            print("Please set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)


if __name__ == "__main__":
    cli()
