# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# utility functions to convert parquet and trace files back and forth

import hashlib
import logging
import os
from collections import defaultdict
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from BackendBench.data_loaders import _load_from_trace
from BackendBench.scripts.dataset_filters import (
    apply_runtime_filter,
    apply_skip_ops_filter,
)
from huggingface_hub import HfApi

DEFAULT_TRACE_URL = "https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/augmented_hf_op_traces.txt"
DEFAULT_PARQUET_URL = "https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/backend_bench_problems.parquet"


"""
Columns for the parquet dataset:
- uuid (int) (hash of op + args)
- op_name (string)
- args (string)
- arg_size (float) (in MB)
- count (int) (number of times this op + set of args was called in real models)
- is_synthetic (boolean) (did we generate this op or is it from a real model)
- included_in_benchmark (boolean)
- why_excluded (list of strings) (empty if included)
- runtime_ms (float) (timings on H100 gpu)
- runnable (bool) (does this op + test work) [we may remove this column later after we solve for special ops]
- in_models (string) (which models did we include this op in) [@TODO add this]
"""

logger = logging.getLogger(__name__)


def _upload_to_hf(file_path: str) -> None:
    """Upload file to GPUMODE/huggingface_op_trace."""
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=Path(file_path).name,
            repo_id="GPUMODE/huggingface_op_trace",
            repo_type="dataset",
        )
        logger.info(f"Uploaded {Path(file_path).name} to Hugging Face")
    except Exception as e:
        logger.warning(f"Failed to upload {Path(file_path).name}: {e}")


def setup_logging(log_level):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s][%(levelname)s][%(filename)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("logs/parquet_trace_converter.log"),
            logging.StreamHandler(),  # Also print to console
        ],
    )


def convert_trace_to_parquet(trace_file, parquet_file, limit: int = None):
    """
    Convert a trace file to a parquet file
    """

    # Load operations using local trace parsing function
    ops = _load_from_trace(trace_file, filter=None, limit=limit)

    # Add additional metadata fields required for the parquet format
    for op in ops:
        op["uuid"] = hashlib.sha256(op["args"].encode() + op["op_name"].encode()).hexdigest()
        op["included_in_benchmark"] = True
        op["why_excluded"] = []
        op["runtime_ms"] = ""
        op["runnable"] = True

    # apply filters
    ops = apply_skip_ops_filter(ops)
    ops = apply_runtime_filter(ops)

    exclusion_dict = defaultdict(lambda: 0)
    exclusion_mapping = defaultdict(lambda: set())
    testable_ops = set()
    all_ops = set()
    for op in ops:
        for reason in op["why_excluded"]:
            exclusion_dict[reason] += 1
            exclusion_mapping[reason].add(op["op_name"])
        if op["included_in_benchmark"]:
            testable_ops.add(op["op_name"])
        all_ops.add(op["op_name"])
    non_testable_ops = all_ops - testable_ops

    for reason, count in exclusion_dict.items():
        logger.info(f"Excluded tests from {count} / {len(ops)} ops due to {reason}")
    logger.info(f"Excluded {len(non_testable_ops)} / {len(all_ops)} ops due to not having tests")
    for reason in exclusion_mapping.keys():
        no_op_set = exclusion_mapping[reason].intersection(non_testable_ops)
        list_str = "\n".join(no_op_set)
        logger.info(
            f"Excluded the following {len(no_op_set)}/{len(all_ops)} ops and input combinations at least partially due to the reason: {reason}:\n {list_str}"
        )

    # Create parquet table with all metadata (formerly "dev" version)
    table = pa.Table.from_pylist(ops)

    # Write parquet file
    pq.write_table(table, parquet_file)

    logger.info(f"Wrote {len(ops)} ops and inputs to {parquet_file}")

    # Log column information for verification
    logger.debug(f"Parquet columns: {table.column_names}")


def convert_parquet_to_trace(parquet_file, trace_file, limit: int = None):
    """
    Convert a parquet file to a trace file
    """
    table = pq.read_table(parquet_file)
    op_inputs = {}

    for row in table.to_pylist():
        formatted_entry = f"cnt: {row['count']}, {row['args']}"

        if row["op_name"] not in op_inputs:
            op_inputs[row["op_name"]] = []
        op_inputs[row["op_name"]].append(formatted_entry)
        if limit:
            op_inputs = op_inputs[:limit]

    # write to trace file
    with open(trace_file, "w") as f:
        for op, args in op_inputs.items():
            f.write(f"Operator: {op}\n")
            for arg in args:
                f.write(f"{arg}\n")
    total_args = sum(len(op_inputs[op]) for op in op_inputs)
    logging.info(f"Wrote {total_args} ops and inputs to {trace_file}")


def _validate_parquet_name(parquet_name: str) -> str:
    """Validate parquet filename. URLs allowed only for inputs."""
    # URLs are allowed only if this is an input file
    if parquet_name.startswith(("http://", "https://")):
        raise click.BadParameter("Output parquet file cannot be a URL")

    if not parquet_name.endswith(".parquet"):
        raise click.BadParameter("Parquet file must end with .parquet suffix")

    # Ensure local files are in datasets directory
    if not parquet_name.startswith("datasets/"):
        parquet_name = os.path.join("datasets", parquet_name)

    return parquet_name


def _validate_trace_file(trace_file: str, is_input: bool = True) -> str:
    """Validate trace file. URLs allowed only for inputs."""
    # URLs are allowed only if this is an input file
    if trace_file.startswith(("http://", "https://")):
        if is_input:
            return trace_file
        else:
            raise click.BadParameter("Output trace file cannot be a URL")

    # For local files, check extension
    if not (trace_file.endswith(".txt") or Path(trace_file).is_dir()):
        raise click.BadParameter("Local trace file must end with .txt or be a directory")

    if Path(trace_file).is_dir() and not is_input:
        raise click.BadParameter("Output trace file cannot be a directory")

    return trace_file


@click.command()
@click.option(
    "--log-level",
    default=os.getenv("LOG_LEVEL", "INFO"),
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
@click.option(
    "--mode",
    default="trace-to-parquet",
    type=click.Choice(["trace-to-parquet", "parquet-to-trace"]),
    help="Conversion mode",
)
@click.option(
    "--trace-file",
    default=DEFAULT_TRACE_URL,
    type=str,
    help="Input trace file: URL (for downloads), local .txt file, or directory. Output trace files cannot be URLs",
)
@click.option(
    "--parquet-name",
    default="backend_bench_problems_runtime_filtered_threshhold_2.parquet",
    type=str,
    help="Parquet filename: URL allowed as input in parquet-to-trace mode, local files in datasets/.",
)
@click.option(
    "--upload-to-hf",
    is_flag=True,
    default=False,
    help="Upload generated parquet files to Hugging Face (GPUMODE/huggingface_op_trace) in trace-to-parquet mode",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Limit the number of operators to convert. (Useful for testing)",
)
def main(log_level, mode, trace_file, parquet_name, upload_to_hf, limit):
    """Convert trace files to parquet format or vice versa."""
    setup_logging(log_level)

    # Create datasets directory
    os.makedirs("datasets", exist_ok=True)

    if mode == "trace-to-parquet":
        # Validate inputs/outputs
        trace_file = _validate_trace_file(trace_file, is_input=True)  # Input: URLs allowed
        parquet_name = _validate_parquet_name(parquet_name)  # Output: URLs not allowed

        logger.info(f"Converting trace file {trace_file} to parquet file {parquet_name}")

        convert_trace_to_parquet(trace_file, parquet_name, limit=limit)
        logger.info("Conversion completed successfully")

        if upload_to_hf:
            # Upload to Hugging Face
            _upload_to_hf(os.path.abspath(parquet_name))

    elif mode == "parquet-to-trace":
        # Validate parquet input (URLs allowed for input in this mode)
        parquet_input = _validate_parquet_name(parquet_name)
        # Validate trace output (URLs not allowed for output)
        trace_output = _validate_trace_file(trace_file, is_input=False)  # Output: URLs not allowed

        logger.info(f"Converting parquet file {parquet_input} to trace file {trace_output}")
        convert_parquet_to_trace(parquet_input, trace_output, limit=limit)
        logger.info("Conversion completed successfully")


if __name__ == "__main__":
    main()
