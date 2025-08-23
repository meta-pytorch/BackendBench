#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Run KernelAgent on multiple PyTorch operators sequentially.
"""

import argparse
import logging
import os
import sys
import subprocess
import shutil
import json
import math
from datetime import datetime
from pathlib import Path

# Add BackendBench to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BackendBench.scripts.setup_operator_directories import clean_op_name_for_directory
from BackendBench.constants import TORCHBENCH_CORE_OPS
from triton_friendly_ops import get_triton_friendly_ops, TRITON_FRIENDLY_OPS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_torchbench_core_ops():
    """Get the list of 77 core TorchBench operators."""
    return TORCHBENCH_CORE_OPS


def get_triton_core_ops():
    """Get Triton-friendly core operators."""
    # Return intersection of core ops and Triton-friendly ops
    return [op for op in TORCHBENCH_CORE_OPS if op in TRITON_FRIENDLY_OPS]


def run_single_op(op, workers, max_rounds, output_base, timestamp, float_only=False):
    """Run KernelAgent on a single operation."""
    run_dir = Path(output_base) / f"kernel_agent_run_{timestamp}"
    
    # Set up environment
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = str(project_root)
    
    # Build command for single op
    cmd = [
        sys.executable,
        "BackendBench/scripts/main.py",
        "--suite", "torchbench",
        "--backend", "kernel_agent_fp16",
        "--ops", op,
        "--kernel-agent-workers", str(workers),
        "--kernel-agent-max-rounds", str(max_rounds)
    ]
    
    logger.info(f"Running KernelAgent for operation: {op}")
    
    # Run the command with timeout per operation
    # Each operation gets up to 5 minutes (300 seconds)
    timeout_seconds = 300
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env
    )
    
    # Capture output and results
    result = {
        "op": op,
        "success": False,
        "correctness": None,
        "performance": None,
        "error": None
    }
    
    for line in process.stdout:
        print(line, end='')
        
        if "✅ KernelAgent succeeded" in line:
            result["success"] = True
        elif "❌ KernelAgent error" in line or "✗ Skipping" in line:
            result["success"] = False
            if ":" in line:
                result["error"] = line.split(":", 1)[1].strip()
        elif "correctness score" in line and "mean pass rate" in line:
            try:
                result["correctness"] = float(line.split(":")[-1].strip())
            except:
                pass
        elif "performance score" in line and "geomean speedup" in line:
            try:
                result["performance"] = float(line.split(":")[-1].strip())
            except:
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
        performance = math.exp(sum(math.log(r["performance"]) for r in successful) / len(successful))
    else:
        performance = None
    
    return {"correctness": correctness, "performance": performance}


def run_kernel_agent_batch(ops_list, workers=4, max_rounds=10, output_base="generated_kernels"):
    """Run KernelAgent on multiple operations sequentially."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_base) / f"kernel_agent_run_{timestamp}"
    
    logger.info(f"Starting KernelAgent batch run with {len(ops_list)} operations")
    logger.info(f"Configuration: {workers} workers, {max_rounds} max rounds")
    logger.info(f"Output will be saved to: {run_dir}")
    
    # Run each op separately to avoid rate limits
    all_results = []
    for i, op in enumerate(ops_list, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing operation {i}/{len(ops_list)}: {op}")
        logger.info(f"{'='*60}")
        
        result = run_single_op(op, workers, max_rounds, output_base, timestamp)
        all_results.append(result)
        
        # Log result
        if result["success"]:
            logger.info(f"✅ {op} succeeded - Correctness: {result['correctness']:.2f}, Performance: {result['performance']:.2f}x")
        else:
            logger.info(f"❌ {op} failed - {result.get('error', 'Unknown error')}")
    
    # Combine scores
    combined_scores = combine_scores(all_results)
    
    return run_dir, combined_scores, all_results


def organize_results(kernel_run_dir, output_base="generated_kernels", scores=None, all_results=None):
    """Organize generated kernels using PR #90 directory structure."""
    if not kernel_run_dir:
        logger.error("No kernel run directory provided")
        return None
    
    # Find the actual kernel agent run directory
    if isinstance(kernel_run_dir, str):
        kernel_run_dir = Path(kernel_run_dir)
    
    if not kernel_run_dir.exists():
        logger.error(f"Kernel run directory does not exist: {kernel_run_dir}")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    organized_dir = Path(output_base) / f"organized_{timestamp}"
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Organizing kernels to: {organized_dir}")
    
    # Find all generated kernel files
    kernel_files = list(kernel_run_dir.glob("*_kernel.py"))
    successful_count = 0
    
    # Create a mapping of op results for detailed READMEs
    op_results = {}
    if all_results:
        for result in all_results:
            op_results[result["op"]] = result
    
    for kernel_file in kernel_files:
        # Extract operation name from filename
        op_name = kernel_file.stem.replace("_kernel", "")
        
        # Clean the operation name for directory
        clean_name = clean_op_name_for_directory(op_name)
        
        # Create operation directory
        op_dir = organized_dir / clean_name
        op_dir.mkdir(exist_ok=True)
        
        # Copy kernel with proper naming convention
        dest_file = op_dir / f"{clean_name}_implementation_v1.py"
        shutil.copy2(kernel_file, dest_file)
        
        # Get specific scores for this operation
        op_result = op_results.get(op_name, {})
        
        # Create README for the operation
        readme_content = f"""# {op_name}

Generated by KernelAgent on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Status
- ✅ Successfully generated and passed BackendBench tests

## Scores
{f"- Correctness: {op_result['correctness']:.2f} (mean pass rate)" if op_result.get('correctness') is not None else "- Correctness: Not measured"}
{f"- Performance: {op_result['performance']:.2f}x (speedup over baseline)" if op_result.get('performance') is not None else "- Performance: Not measured"}

## Implementation
The kernel implementation is in `{clean_name}_implementation_v1.py`.

## Source
Original kernel: {kernel_file}
"""
        (op_dir / "README.md").write_text(readme_content)
        
        successful_count += 1
        logger.info(f"Organized {op_name} -> {op_dir}")
    
    # Create summary README
    summary_content = f"""# KernelAgent Generated Kernels

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Total operations attempted: {len(all_results) if all_results else 0}
- Successfully generated: {successful_count}
- Success rate: {successful_count/len(all_results)*100:.1f}% if all_results else 0%

## Overall Scores
{f"- Correctness: {scores['correctness']:.2f} (mean pass rate)" if scores and scores.get('correctness') is not None else "- Correctness: Not measured"}
{f"- Performance: {scores['performance']:.2f}x (geomean speedup)" if scores and scores.get('performance') is not None else "- Performance: Not measured"}

## Individual Results
"""
    
    if all_results:
        for result in all_results:
            status = "✅" if result["success"] else "❌"
            summary_content += f"\n### {result['op']} {status}\n"
            if result["success"]:
                summary_content += f"- Correctness: {result['correctness']:.2f}\n" if result.get('correctness') else ""
                summary_content += f"- Performance: {result['performance']:.2f}x\n" if result.get('performance') else ""
            else:
                summary_content += f"- Error: {result.get('error', 'Unknown error')}\n"
    
    summary_content += f"""
## Directory Structure
Each operation has its own directory following the PR #90 convention:
- `<op_name>/` - Operation directory
  - `README.md` - Operation details and scores
  - `<op_name>_implementation_v1.py` - Kernel implementation

## Usage with DirectoryBackend
```bash
python BackendBench/scripts/main.py --suite torchbench --backend directory --ops-directory {organized_dir}
```
"""
    (organized_dir / "README.md").write_text(summary_content)
    
    # Save detailed results to JSON
    if scores or all_results:
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "total_operations": len(all_results) if all_results else 0,
            "successful_operations": successful_count,
            "overall_scores": scores,
            "individual_results": all_results,
            "configuration": {
                "workers": 4,
                "max_rounds": 10
            }
        }
        with open(organized_dir / "results.json", "w") as f:
            json.dump(results_data, f, indent=2)
    
    logger.info(f"Organization complete: {successful_count} kernels organized")
    return organized_dir


def main():
    parser = argparse.ArgumentParser(description="Run KernelAgent on PyTorch operators")
    parser.add_argument(
        "--ops",
        type=str,
        help="Comma-separated list of operations (default: 77 core ops)",
        default=None
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers per operation (default: 4)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum refinement rounds per operation (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_kernels",
        help="Base output directory (default: generated_kernels)"
    )
    parser.add_argument(
        "--triton-friendly",
        action="store_true",
        help="Only test Triton-friendly operations that work well with float dtypes"
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
    else:
        ops_list = get_torchbench_core_ops()
        logger.info(f"Running {len(ops_list)} core TorchBench operations")
    
    # Run KernelAgent batch
    kernel_run_dir, scores, all_results = run_kernel_agent_batch(
        ops_list,
        workers=args.workers,
        max_rounds=args.max_rounds,
        output_base=args.output_dir
    )
    
    if kernel_run_dir:
        # Organize results
        organized_dir = organize_results(kernel_run_dir, args.output_dir, scores=scores, all_results=all_results)
        
        if organized_dir:
            logger.info("=" * 80)
            logger.info("Run completed successfully!")
            logger.info(f"Organized kernels: {organized_dir}")
            if scores and scores.get("correctness") is not None:
                logger.info(f"Overall Correctness: {scores['correctness']:.2f}")
            if scores and scores.get("performance") is not None:
                logger.info(f"Overall Performance: {scores['performance']:.2f}x")
            logger.info("=" * 80)


if __name__ == "__main__":
    main()