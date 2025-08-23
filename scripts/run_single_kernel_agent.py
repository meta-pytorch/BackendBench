#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Run KernelAgent on a single PyTorch operator.
"""

import argparse
import logging
import os
import sys
import subprocess
import shutil
import json
from datetime import datetime
from pathlib import Path

# Add BackendBench to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BackendBench.scripts.setup_operator_directories import clean_op_name_for_directory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_op(op, workers=4, max_rounds=10, output_base="generated_kernels"):
    """Run KernelAgent on a single operation and return results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_base) / f"kernel_agent_run_{op}_{timestamp}"
    
    # Set up environment
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = str(project_root)
    
    # Build command
    cmd = [
        sys.executable,
        "BackendBench/scripts/main.py",
        "--suite", "torchbench",
        "--backend", "kernel_agent",
        "--ops", op,
        "--kernel-agent-workers", str(workers),
        "--kernel-agent-max-rounds", str(max_rounds)
    ]
    
    logger.info(f"Starting KernelAgent for operation: {op}")
    logger.info(f"Configuration: {workers} workers, {max_rounds} max rounds")
    logger.info(f"Output directory: {run_dir}")
    
    # Run the command
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
        "error": None,
        "variants": []
    }
    
    current_variant = None
    
    for line in process.stdout:
        print(line, end='')
        
        # Track which variant is being processed
        if "] " in line and " - KernelAgent Generation" in line:
            parts = line.split("] ", 1)
            if len(parts) > 1:
                variant_name = parts[1].split(" - ")[0].strip()
                current_variant = variant_name
        
        # Track success/failure per variant
        if current_variant:
            if "✅ KernelAgent succeeded" in line:
                result["variants"].append({"name": current_variant, "status": "success"})
                result["success"] = True  # At least one variant succeeded
            elif "❌ KernelAgent error" in line or "✗ Skipping" in line:
                error_msg = line.split(":", 1)[1].strip() if ":" in line else "Unknown error"
                result["variants"].append({"name": current_variant, "status": "failed", "error": error_msg})
        
        # Capture final scores
        if "correctness score" in line and "mean pass rate" in line:
            try:
                result["correctness"] = float(line.split(":")[-1].strip())
            except:
                pass
        elif "performance score" in line and "geomean speedup" in line:
            try:
                result["performance"] = float(line.split(":")[-1].strip())
            except:
                pass
    
    process.wait()
    
    if process.returncode != 0 and not result["success"]:
        result["error"] = f"Process exited with code {process.returncode}"
    
    # Save result summary
    result_file = run_dir / "result_summary.json"
    if run_dir.exists():
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result summary saved to: {result_file}")
    
    return result, run_dir


def organize_results(run_dir, result, output_base="generated_kernels"):
    """Organize generated kernels using PR #90 directory structure."""
    if not run_dir.exists():
        logger.error(f"Run directory does not exist: {run_dir}")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    organized_dir = Path(output_base) / f"organized_{timestamp}"
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Organizing kernels to: {organized_dir}")
    
    # Find all generated kernel files
    kernel_files = list(run_dir.glob("*_kernel.py"))
    successful_count = 0
    
    for kernel_file in kernel_files:
        # Extract operation name
        op_name = kernel_file.stem.replace("_kernel", "")
        clean_name = clean_op_name_for_directory(op_name)
        
        # Create operation directory
        op_dir = organized_dir / clean_name
        op_dir.mkdir(exist_ok=True)
        
        # Copy kernel
        dest_file = op_dir / f"{clean_name}_implementation_v1.py"
        shutil.copy2(kernel_file, dest_file)
        
        # Create README with scores
        readme_content = f"""# {op_name}

Generated by KernelAgent on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Status
- ✅ Successfully generated and passed BackendBench tests

## Scores
{f"- Correctness: {result['correctness']:.2f} (mean pass rate)" if result.get('correctness') is not None else "- Correctness: Not measured"}
{f"- Performance: {result['performance']:.2f}x (speedup over baseline)" if result.get('performance') is not None else "- Performance: Not measured"}

## Variants Attempted
"""
        for variant in result.get("variants", []):
            status_icon = "✅" if variant["status"] == "success" else "❌"
            readme_content += f"- {status_icon} {variant['name']}"
            if variant.get("error"):
                readme_content += f" - {variant['error']}"
            readme_content += "\n"
        
        readme_content += f"""
## Implementation
The kernel implementation is in `{clean_name}_implementation_v1.py`.

## Source
Original kernel: {kernel_file}
"""
        (op_dir / "README.md").write_text(readme_content)
        
        successful_count += 1
        logger.info(f"Organized {op_name} -> {op_dir}")
    
    # Save overall summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "operation": result["op"],
        "successful_kernels": successful_count,
        "correctness_score": result.get("correctness"),
        "performance_score": result.get("performance"),
        "variants": result.get("variants", []),
        "configuration": {
            "workers": 4,
            "max_rounds": 10
        }
    }
    
    with open(organized_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Organization complete: {successful_count} kernels organized")
    return organized_dir


def main():
    parser = argparse.ArgumentParser(description="Run KernelAgent on a single PyTorch operator")
    parser.add_argument(
        "op",
        type=str,
        help="The operator to generate a kernel for (e.g., relu, add, mul)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum refinement rounds (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_kernels",
        help="Base output directory (default: generated_kernels)"
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize results after generation"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("ERROR: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Run KernelAgent
    result, run_dir = run_single_op(
        args.op,
        workers=args.workers,
        max_rounds=args.max_rounds,
        output_base=args.output_dir
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Operation: {result['op']}")
    print(f"Success: {result['success']}")
    
    if result["success"]:
        print(f"Correctness: {result['correctness']:.2f}" if result['correctness'] else "Correctness: Not measured")
        print(f"Performance: {result['performance']:.2f}x" if result['performance'] else "Performance: Not measured")
        
        if args.organize:
            organized_dir = organize_results(run_dir, result, args.output_dir)
            if organized_dir:
                print(f"\nOrganized results: {organized_dir}")
    else:
        print(f"Error: {result.get('error', 'Failed to generate kernel')}")
    
    print("\nVariants attempted:")
    for variant in result.get("variants", []):
        status_icon = "✅" if variant["status"] == "success" else "❌"
        print(f"  {status_icon} {variant['name']}", end="")
        if variant.get("error"):
            print(f" - {variant['error']}")
        else:
            print()
    
    print("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()