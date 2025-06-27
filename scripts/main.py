import logging
import sys

import BackendBench.backends as backends
import BackendBench.eval as eval
import click
import torch
from BackendBench.opinfo_suite import OpInfoTestSuite
from BackendBench.suite import SmokeTestSuite

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
    type=click.Choice(["aten", "flag_gems"]),
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


if __name__ == "__main__":
    cli()
