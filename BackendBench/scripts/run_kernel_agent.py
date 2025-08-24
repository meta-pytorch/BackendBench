#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Run KernelAgent on PyTorch operators (single or multiple).

This script can run KernelAgent on:
- A single operation: --ops "relu"
- Multiple operations: --ops "relu,sigmoid,tanh"
- All core ops: (default)
- Triton-friendly ops: --triton-friendly
"""

import argparse
import logging
import os
import sys
import subprocess
import math
from pathlib import Path

from ..constants import TORCHBENCH_CORE_OPS
from .triton_friendly_ops import TRITON_FRIENDLY_OPS_EXPANDED as TRITON_FRIENDLY_OPS

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_torchbench_core_ops():
    """Get the list of 77 core TorchBench operators."""
    return TORCHBENCH_CORE_OPS


def get_triton_core_ops():
    """Get Triton-friendly core operators."""
    # Return intersection of core ops and Triton-friendly ops
    return [op for op in TORCHBENCH_CORE_OPS if op in TRITON_FRIENDLY_OPS]


def get_triton_capable_core_ops():
    """Get Triton-capable core operators (require more engineering)."""
    from .triton_friendly_ops import TRITON_CAPABLE_OPS
    return [op for op in TORCHBENCH_CORE_OPS if op in TRITON_CAPABLE_OPS]


def run_single_op(op, workers, max_rounds, output_base, float_only=False):
    """Run KernelAgent on a single operation."""

    # Set up environment
    env = os.environ.copy()
    # Script is now in BackendBench/scripts/, so go up 2 levels to get project root
    project_root = Path(__file__).parent.parent.parent
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(project_root)

    # Build command for single op
    cmd = [
        sys.executable,
        "BackendBench/scripts/main.py",
        "--suite",
        "torchbench",
        "--backend",
        "kernel_agent",
        "--ops",
        op,
        "--kernel-agent-workers",
        str(workers),
        "--kernel-agent-max-rounds",
        str(max_rounds),
        "--filter-fp16-bf16",  # Always filter to FP16/BF16 for better correctness
    ]

    logger.info(f"Running KernelAgent for operation: {op}")

    # Run the command with timeout per operation
    # Each operation gets up to 5 minutes (300 seconds)
    timeout_seconds = 300

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env
    )

    # Capture output and results
    result = {"op": op, "success": False, "correctness": None, "performance": None, "error": None}

    for line in process.stdout:
        print(line, end="")

        if "✅ KernelAgent succeeded" in line:
            result["success"] = True
        elif "❌ KernelAgent error" in line or "✗ Skipping" in line:
            result["success"] = False
            if ":" in line:
                result["error"] = line.split(":", 1)[1].strip()
        elif "correctness score" in line and "mean pass rate" in line:
            try:
                result["correctness"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "performance score" in line and "geomean speedup" in line:
            try:
                result["performance"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    # Wait with timeout
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(f"Operation {op} timed out after {timeout_seconds} seconds")
        process.kill()
        result["error"] = f"Timed out after {timeout_seconds} seconds"
        result["success"] = False

    return result


def combine_scores(results):
    """Combine scores from multiple single-op runs."""
    successful = [r for r in results if r["success"] and r["correctness"] is not None]

    if not successful:
        return {"correctness": None, "performance": None}

    # Average correctness scores
    correctness = sum(r["correctness"] for r in successful) / len(successful)

    # Geometric mean for performance scores
    if all(r["performance"] is not None for r in successful):
        performance = math.exp(
            sum(math.log(r["performance"]) for r in successful) / len(successful)
        )
    else:
        performance = None

    return {"correctness": correctness, "performance": performance}


def run_kernel_agent_batch(ops_list, workers=4, max_rounds=10, output_base="generated_kernels"):
    """Run KernelAgent on multiple operations sequentially."""
    logger.info(f"Starting KernelAgent batch run with {len(ops_list)} operations")
    logger.info(f"Configuration: {workers} workers, {max_rounds} max rounds")
    logger.info(f"Output will be saved to: {output_base} (PR #90 structure)")

    # Run each op separately to avoid rate limits
    all_results = []
    for i, op in enumerate(ops_list, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing operation {i}/{len(ops_list)}: {op}")
        logger.info(f"{'=' * 60}")

        result = run_single_op(op, workers, max_rounds, output_base)
        all_results.append(result)

        # Log result
        if result["success"]:
            logger.info(
                f"✅ {op} succeeded - Correctness: {result['correctness']:.2f}, Performance: {result['performance']:.2f}x"
            )
        else:
            logger.info(f"❌ {op} failed - {result.get('error', 'Unknown error')}")

    # Combine scores
    combined_scores = combine_scores(all_results)

    return combined_scores, all_results


def main():
    parser = argparse.ArgumentParser(description="Run KernelAgent on PyTorch operators")
    parser.add_argument(
        "--ops",
        type=str,
        help="Comma-separated list of operations (default: 77 core ops)",
        default=None,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers per operation (default: 4)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum refinement rounds per operation (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_kernels",
        help="Base output directory (default: generated_kernels)",
    )
    parser.add_argument(
        "--triton-friendly",
        action="store_true",
        help="Only test Triton-friendly operations (easy wins with good performance)",
    )
    parser.add_argument(
        "--triton-capable",
        action="store_true",
        help="Test Triton-capable operations (require careful engineering)",
    )

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("ERROR: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Determine operations to run
    if args.ops:
        ops_list = [op.strip() for op in args.ops.split(",")]
        logger.info(f"Running {len(ops_list)} specified operations")
    elif args.triton_friendly:
        ops_list = get_triton_core_ops()
        logger.info(f"Running {len(ops_list)} Triton-friendly core operations")
        logger.info(f"Operations: {', '.join(ops_list[:10])}{'...' if len(ops_list) > 10 else ''}")
    elif args.triton_capable:
        ops_list = get_triton_capable_core_ops()
        logger.info(f"Running {len(ops_list)} Triton-capable core operations (require careful engineering)")
        logger.info(f"Operations: {', '.join(ops_list[:10])}{'...' if len(ops_list) > 10 else ''}")
    else:
        ops_list = get_torchbench_core_ops()
        logger.info(f"Running {len(ops_list)} core TorchBench operations")

    # Run KernelAgent batch
    scores, all_results = run_kernel_agent_batch(
        ops_list, workers=args.workers, max_rounds=args.max_rounds, output_base=args.output_dir
    )

    logger.info("=" * 80)
    logger.info("Run completed successfully!")
    logger.info(f"Kernels saved to: {args.output_dir} (PR #90 structure)")
    if scores and scores.get("correctness") is not None:
        logger.info(f"Overall Correctness: {scores['correctness']:.2f}")
    if scores and scores.get("performance") is not None:
        logger.info(f"Overall Performance: {scores['performance']:.2f}x")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
