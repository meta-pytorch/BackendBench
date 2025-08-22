# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pyarrow.parquet as pq
import requests
import torch
import tqdm
from BackendBench.data_loaders import _load_from_trace
from BackendBench.utils import deserialize_args
import gc

def safe_cleanup_memory_and_gpu():
    """Safe cleanup that works with or without CUDA."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

from triton.testing import do_bench
TRITON_AVAILABLE = True


SKIP_OPERATORS = [
    "embedding",
    "scatter",
    "gather",
    "index",
    "nll_loss",
    "im2col_backward",
    "col2im_backward",
    "native_layer_norm_backward",
    "upsample_nearest2d_backward.vec",
    "upsample_bilinear2d_backward.vec",
    "_cudnn_rnn_backward.default",
    "_fft_c2c.default",
]

DEFAULT_PARQUET_URL = "https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/backend_bench_problems.parquet"
DEFAULT_TRACE_URL = "https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/augmented_hf_op_traces.txt"


def setup_logging(log_level):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s][%(levelname)s][%(filename)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("logs/runtime_histogram.log"),
            logging.StreamHandler(),
        ],
    )


def filter_and_measure_ops(ops, baseline_ms, device='cuda', include_synthetic=True):
    """Filter ops based on skip list, measure runtime for both synthetic and non-synthetic ops."""
    filtered_ops = []
    synthetic_ops = []
    runtime_data = []
    synthetic_runtime_data = []
    op_runtime_map = defaultdict(list)
    synthetic_op_runtime_map = defaultdict(list)
    failed_ops = []
    skipped_ops = []
    
    for op in tqdm.tqdm(ops, desc="Filtering and measuring ops"):
        # Apply skip ops filter
        if any(skip_op in op["op_name"] for skip_op in SKIP_OPERATORS):
            skipped_ops.append({
                "op_name": op["op_name"],
                "reason": "In skip list"
            })
            continue
            
        # Try to measure runtime
        args, kwargs = None, None
        try:
            args, kwargs = deserialize_args(op["args"])
            
            op_name = op["op_name"]
            op_func = eval(f"torch.ops.{op_name}")
            runtime_ms = do_bench(lambda: op_func(*args, **kwargs), warmup=25, rep=100)
            
            # Calculate relative runtime
            relative_runtime = runtime_ms / baseline_ms
            
            # Separate synthetic and non-synthetic ops
            is_synthetic = op.get("is_synthetic", False)
            op_data = {
                "op_name": op_name,
                "runtime_ms": runtime_ms,
                "relative_runtime": relative_runtime,
                "args": op["args"],
                "is_synthetic": is_synthetic,
            }
            
            if is_synthetic:
                if include_synthetic:
                    synthetic_ops.append(op_data)
                    synthetic_runtime_data.append(relative_runtime)
                    synthetic_op_runtime_map[op_name].append(relative_runtime)
                else:
                    skipped_ops.append({
                        "op_name": op["op_name"],
                        "reason": "Synthetic op (excluded by flag)"
                    })
            else:
                filtered_ops.append(op_data)
                runtime_data.append(relative_runtime)
                op_runtime_map[op_name].append(relative_runtime)
            
        except Exception as e:
            failed_ops.append({
                "op_name": op.get("op_name", "unknown"),
                "error": str(e)
            })
            logging.debug(f"Failed to run {op.get('op_name', 'unknown')}: {e}")
        finally:
            if args:
                del args
            if kwargs:
                del kwargs
            safe_cleanup_memory_and_gpu()
    
    return filtered_ops, synthetic_ops, runtime_data, synthetic_runtime_data, op_runtime_map, synthetic_op_runtime_map, failed_ops, skipped_ops


def create_histogram(runtime_data, output_file=None, bins=30, log_scale=True, title="RUNTIME DISTRIBUTION HISTOGRAM"):
    """Create and display ASCII histogram of runtime distribution in terminal."""
    if not runtime_data:
        logging.warning("No runtime data to plot")
        return
    
    # Filter out extreme outliers for better visualization
    data = np.array(runtime_data)
    percentile_99 = np.percentile(data, 99)
    data_filtered = data[data <= percentile_99]
    
    # Create histogram
    if log_scale:
        # Use log bins for better distribution visualization
        min_val = max(data_filtered.min(), 0.001)  # Avoid log(0)
        max_val = data_filtered.max()
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins + 1)
    else:
        bin_edges = np.linspace(data_filtered.min(), data_filtered.max(), bins + 1)
    
    hist, edges = np.histogram(data_filtered, bins=bin_edges)
    
    # ASCII histogram
    max_count = hist.max()
    bar_width = 40  # Width of the histogram bars
    
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Relative Runtime {'(log scale) ' if log_scale else ''}- relative to torch.empty(0, device='cuda')")
    print("-" * 80)
    
    for i in range(len(hist)):
        # Calculate bar length
        if max_count > 0:
            bar_length = int((hist[i] / max_count) * bar_width)
        else:
            bar_length = 0
        
        # Format bin range
        if log_scale:
            bin_label = f"{edges[i]:7.3f}x - {edges[i+1]:7.3f}x"
        else:
            bin_label = f"{edges[i]:7.2f}x - {edges[i+1]:7.2f}x"
        
        # Create bar
        bar = "█" * bar_length + "░" * (bar_width - bar_length)
        
        # Print with count
        print(f"{bin_label} │ {bar} │ {hist[i]:5d}")
    
    print("-" * 80)
    # Threshold analysis
    ops_below_1x = np.sum(data < 1.0)
    ops_below_1_5x = np.sum(data < 1.5)
    ops_below_2x = np.sum(data < 2.0)
    ops_above_2x = np.sum(data >= 2.0)
    
    # Statistics
    stats_text = f'Total ops: {len(runtime_data)}\n'
    stats_text += f'Mean: {np.mean(data):.2f}x\n'
    stats_text += f'Median: {np.median(data):.2f}x\n'
    stats_text += f'Min: {np.min(data):.2f}x\n'
    stats_text += f'Max: {np.max(data):.2f}x\n'
    stats_text += f'99th percentile: {percentile_99:.2f}x\n'
    stats_text += f'\nThreshold Analysis (vs torch.empty baseline):\n'
    stats_text += f'Ops < 1.0x threshold: {ops_below_1x} ({100*ops_below_1x/len(data):.1f}%)\n'
    stats_text += f'Ops < 1.5x threshold: {ops_below_1_5x} ({100*ops_below_1_5x/len(data):.1f}%)\n'
    stats_text += f'Ops < 2.0x threshold: {ops_below_2x} ({100*ops_below_2x/len(data):.1f}%)\n'
    stats_text += f'Ops ≥ 2.0x threshold: {ops_above_2x} ({100*ops_above_2x/len(data):.1f}%)'
    
    return stats_text


def print_top_ops(op_runtime_map, n=20):
    """Print the top N slowest and fastest operations."""
    # Calculate mean runtime for each op
    op_means = {op: np.mean(runtimes) for op, runtimes in op_runtime_map.items()}
    
    # Sort by runtime
    sorted_ops = sorted(op_means.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {n} slowest operations (relative to baseline):")
    print("-" * 60)
    for i, (op, runtime) in enumerate(sorted_ops[:n], 1):
        count = len(op_runtime_map[op])
        print(f"{i:3}. {op:40} {runtime:8.2f}x (n={count})")
    
    print(f"\nTop {n} fastest operations (relative to baseline):")
    print("-" * 60)
    for i, (op, runtime) in enumerate(sorted_ops[-n:][::-1], 1):
        count = len(op_runtime_map[op])
        print(f"{i:3}. {op:40} {runtime:8.2f}x (n={count})")
    max_runtimes = {k: max(v) for k, v in op_runtime_map.items()}
    max_runtimes = np.array([v for k, v in max_runtimes.items()])
    ops_below_1x = np.sum(max_runtimes < 1.0)
    ops_below_1_5x = np.sum(max_runtimes < 1.5)
    ops_below_2x = np.sum(max_runtimes < 2.0)
    ops_above_2x = np.sum(max_runtimes >= 2.0)
    print(f"\nChecking if ops have at least one test above threshold (vs torch.empty baseline):\n")
    print(f'Ops < 1.0x threshold: {ops_below_1x}')
    print(f'Ops < 1.5x threshold: {ops_below_1_5x}')
    print(f'Ops < 2.0x threshold: {ops_below_2x}')
    print(f'Ops ≥ 2.0x threshold: {ops_above_2x}')




@click.command()
@click.option(
    "--log-level",
    default=os.getenv("LOG_LEVEL", "INFO"),
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
@click.option(
    "--source",
    default=DEFAULT_PARQUET_URL,
    type=str,
    help="Input source: parquet file, trace file, or URL",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Limit the number of operators to process (useful for testing)",
)
@click.option(
    "--bins",
    default=30,
    type=int,
    help="Number of bins for histogram",
)
@click.option(
    "--log-scale/--no-log-scale",
    default=True,
    help="Use log scale for x-axis",
)
@click.option(
    "--top-n",
    default=20,
    type=int,
    help="Number of top/bottom ops to display",
)
@click.option(
    "--include-synthetic/--no-synthetic",
    default=True,
    help="Include synthetic ops in the analysis (default: True)",
)
def main(log_level, source, limit, bins, log_scale, top_n, include_synthetic):
    """Generate histogram distribution of operation runtimes relative to baseline."""
    setup_logging(log_level)
    
    logging.info("Starting runtime histogram generation")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Using CPU for benchmarking (results may vary).")
    
    # Measure baseline runtime
    logging.info(f"Measuring baseline runtime on {device}...")
    baseline_op = lambda: torch.empty(0, device=device)
    baseline_ms = do_bench(baseline_op, warmup=25, rep=100)
    logging.info(f"Baseline runtime: {baseline_ms:.6f} ms")
    
    # Load operations
    logging.info(f"Loading operations from {source}")
    
    # Determine format and load
    if source.endswith(".parquet"):
        # Handle URLs for parquet files
        if source.startswith(("http://", "https://")):
            logging.info("Downloading parquet file...")
            response = requests.get(source)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            table = pq.read_table(tmp_path)
            ops = table.to_pylist()
            # Clean up temp file
            os.unlink(tmp_path)
        else:
            table = pq.read_table(source)
            ops = table.to_pylist()
    else:
        # Assume trace format
        ops = _load_from_trace(source, filter=None, limit=limit)
        # Add metadata fields
        for op in ops:
            op["uuid"] = hashlib.sha256(op["args"].encode() + op["op_name"].encode()).hexdigest()
    
    if limit:
        ops = ops[:limit]
    
    logging.info(f"Loaded {len(ops)} operations")
    
    # Filter and measure operations
    filtered_ops, synthetic_ops, runtime_data, synthetic_runtime_data, op_runtime_map, synthetic_op_runtime_map, failed_ops, skipped_ops = filter_and_measure_ops(ops, baseline_ms, device, include_synthetic)
    
    logging.info(f"Successfully measured {len(filtered_ops)} non-synthetic operations")
    logging.info(f"Successfully measured {len(synthetic_ops)} synthetic operations")
    logging.info(f"Failed to run {len(failed_ops)} operations")
    logging.info(f"Skipped {len(skipped_ops)} operations")
    
    if not filtered_ops and not synthetic_ops:
        logging.error("No operations passed filters and measurements")
        return
    
    # Prepare output capture
    import io
    import sys
    output_buffer = io.StringIO()
    
    # Create a wrapper for dual output
    class DualOutput:
        def __init__(self, file1, file2):
            self.file1 = file1
            self.file2 = file2
        
        def write(self, data):
            self.file1.write(data)
            self.file2.write(data)
        
        def flush(self):
            self.file1.flush()
            self.file2.flush()
    
    # Save original stdout and redirect to dual output
    original_stdout = sys.stdout
    sys.stdout = DualOutput(original_stdout, output_buffer)
    
    # Create histograms for non-synthetic ops
    if runtime_data:
        stats = create_histogram(runtime_data, bins=bins, log_scale=log_scale, 
                                title="NON-SYNTHETIC OPS RUNTIME DISTRIBUTION HISTOGRAM")
        
        # Print statistics for non-synthetic ops
        print("\n" + "=" * 60)
        print("Non-Synthetic Ops Runtime Statistics")
        print("=" * 60)
        print(stats)
        
        # Print top non-synthetic operations
        print("\n" + "=" * 60)
        print("TOP NON-SYNTHETIC OPERATIONS")
        print("=" * 60)
        print_top_ops(op_runtime_map, n=top_n)
    
    # Create histograms for synthetic ops
    if synthetic_runtime_data:
        synthetic_stats = create_histogram(synthetic_runtime_data, bins=bins, log_scale=log_scale,
                                          title="SYNTHETIC OPS RUNTIME DISTRIBUTION HISTOGRAM")
        
        # Print statistics for synthetic ops
        print("\n" + "=" * 60)
        print("Synthetic Ops Runtime Statistics")
        print("=" * 60)
        print(synthetic_stats)
        
        # Print top synthetic operations
        print("\n" + "=" * 60)
        print("TOP SYNTHETIC OPERATIONS")
        print("=" * 60)
        print_top_ops(synthetic_op_runtime_map, n=top_n)
    
    # Print failed operations
    if failed_ops:
        print("\n" + "=" * 60)
        print(f"Failed Operations ({len(failed_ops)} total)")
        print("=" * 60)
        # Group failed ops by error message
        error_groups = defaultdict(list)
        for failed_op in failed_ops:
            error_groups[failed_op["error"]].append(failed_op["op_name"])
        
        for error, op_names in sorted(error_groups.items(), key=lambda x: -len(x[1])):
            print(f"\nError: {error[:100]}...")
            print(f"Affected ops ({len(op_names)}): {', '.join(op_names[:5])}")
            if len(op_names) > 5:
                print(f"  ... and {len(op_names) - 5} more")
    
    # Restore original stdout
    sys.stdout = original_stdout
    
    # Save output to file
    output_file = "runtime_histogram_output.txt"
    with open(output_file, "w") as f:
        f.write(output_buffer.getvalue())
    
    print(f"\n[Output saved to {output_file}]")
    logging.info(f"Output saved to {output_file}")
    logging.info("Runtime histogram generation complete")


if __name__ == "__main__":
    main()
