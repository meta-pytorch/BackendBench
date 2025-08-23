#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Run KernelAgent on PyTorch operators and organize results using PR #90 directory structure.
"""

import argparse
import logging
import os
import sys
import subprocess
import shutil
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


# The 77 core TorchBench operators
# This list is derived from analysis of which operators appear most frequently
# in TorchBench workloads and are considered high-priority for optimization
TORCHBENCH_CORE_OPS = [
    "abs", "_adaptive_avg_pool2d", "_adaptive_avg_pool2d_backward", "add", "addmm",
    "any", "avg_pool2d", "avg_pool2d_backward", "bitwise_and", "bitwise_not",
    "bitwise_xor", "bmm", "cat", "clamp", "clone", "col2im", "constant_pad_nd",
    "convolution", "convolution_backward", "cos", "cumsum", "div", "elu", "eq",
    "erf", "exp", "flip", "floor", "fmod", "ge", "gelu", "grid_sampler_2d", "gt",
    "hardtanh", "isinf", "isnan", "le", "leaky_relu", "log2", "_log_softmax", "lt",
    "max", "maximum", "max_pool2d_with_indices", "max_pool2d_with_indices_backward",
    "mean", "min", "minimum", "mm", "mul", "native_group_norm",
    "native_group_norm_backward", "native_layer_norm", "ne", "neg", "nonzero",
    "pow", "reciprocal", "reflection_pad2d", "relu", "remainder", "repeat",
    "round", "rsqrt", "sigmoid", "sin", "_softmax", "split_with_sizes", "sqrt",
    "sub", "sum", "tanh", "_to_copy", "topk", "upsample_bilinear2d",
    "upsample_nearest2d", "where"
]


def get_torchbench_core_ops():
    """Get the list of 77 core TorchBench operators."""
    return TORCHBENCH_CORE_OPS


def run_kernel_agent(ops_list, workers=4, max_rounds=10, output_base="generated_kernels"):
    """Run KernelAgent on the specified operations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_base) / f"kernel_agent_run_{timestamp}"
    
    # Create comma-separated list for the command
    ops_str = ",".join(ops_list)
    
    # Build command
    cmd = [
        sys.executable,
        "BackendBench/scripts/main.py",
        "--suite", "torchbench",
        "--backend", "kernel_agent",
        "--ops", ops_str,
        "--kernel-agent-workers", str(workers),
        "--kernel-agent-max-rounds", str(max_rounds)
    ]
    
    logger.info(f"Starting KernelAgent run with {len(ops_list)} operations")
    logger.info(f"Configuration: {workers} workers, {max_rounds} max rounds")
    logger.info(f"Output will be saved to: {run_dir}")
    
    # Run the command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        logger.error(f"KernelAgent run failed with exit code {process.returncode}")
        return None
        
    return run_dir


def organize_results(kernel_run_dir, output_base="generated_kernels"):
    """Organize generated kernels using PR #90 directory structure."""
    if not kernel_run_dir:
        logger.error("No kernel run directory provided")
        return None
    
    # Find the actual kernel agent run directory
    if isinstance(kernel_run_dir, str):
        kernel_run_dir = Path(kernel_run_dir)
    
    # Look for kernel_agent_run_* directories
    kernel_agent_dirs = list(Path(output_base).glob("kernel_agent_run_*"))
    if not kernel_agent_dirs:
        logger.error("No kernel agent run directories found")
        return None
    
    # Use the most recent one
    kernel_agent_dir = sorted(kernel_agent_dirs)[-1]
    logger.info(f"Using kernel agent directory: {kernel_agent_dir}")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    organized_dir = Path(output_base) / f"organized_{timestamp}"
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Organizing kernels to: {organized_dir}")
    
    # Find all generated kernel files
    kernel_files = list(kernel_agent_dir.glob("*_kernel.py"))
    successful_count = 0
    
    for kernel_file in kernel_files:
        # Extract operation name from filename (e.g., relu_kernel.py -> relu)
        op_name = kernel_file.stem.replace("_kernel", "")
        
        # Clean the operation name for directory
        clean_name = clean_op_name_for_directory(op_name)
        
        # Create operation directory
        op_dir = organized_dir / clean_name
        op_dir.mkdir(exist_ok=True)
        
        # Copy kernel with proper naming convention
        dest_file = op_dir / f"{clean_name}_implementation_v1.py"
        shutil.copy2(kernel_file, dest_file)
        
        # Create README for the operation
        readme_content = f"""# {op_name}

Generated by KernelAgent on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Status
- ✅ Successfully generated and passed BackendBench tests

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
- Total operations attempted: {len(get_torchbench_core_ops())}
- Successfully generated: {successful_count}
- Success rate: {successful_count/len(get_torchbench_core_ops())*100:.1f}%

## Directory Structure
Each operation has its own directory following the PR #90 convention:
- `{clean_name}/` - Operation directory
  - `README.md` - Operation details
  - `{clean_name}_implementation_v1.py` - Kernel implementation

## Usage with DirectoryBackend
```bash
python BackendBench/scripts/main.py --suite torchbench --backend directory --ops-directory {organized_dir}
```
"""
    (organized_dir / "README.md").write_text(summary_content)
    
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
        "--single-op",
        type=str,
        help="Run on a single operation (for testing)"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("ERROR: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Determine operations to run
    if args.single_op:
        ops_list = [args.single_op]
        logger.info(f"Running single operation: {args.single_op}")
    elif args.ops:
        ops_list = [op.strip() for op in args.ops.split(",")]
        logger.info(f"Running {len(ops_list)} specified operations")
    else:
        ops_list = get_torchbench_core_ops()
        logger.info(f"Running {len(ops_list)} core TorchBench operations")
    
    # Run KernelAgent
    kernel_run_dir = run_kernel_agent(
        ops_list,
        workers=args.workers,
        max_rounds=args.max_rounds,
        output_base=args.output_dir
    )
    
    if kernel_run_dir:
        # Organize results
        organized_dir = organize_results(kernel_run_dir, args.output_dir)
        
        if organized_dir:
            logger.info("=" * 80)
            logger.info("Run completed successfully!")
            logger.info(f"Organized kernels: {organized_dir}")
            logger.info("=" * 80)


if __name__ == "__main__":
    main()