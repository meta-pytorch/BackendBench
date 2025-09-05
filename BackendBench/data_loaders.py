# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared data loading utilities for reading trace and parquet files.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pyarrow.parquet as pq

import requests
import torch
from datasets import load_dataset
from tqdm import tqdm


# constants for downloading the test set from huggingface
# you can explore the dataset here
# https://huggingface.co/datasets/GPUMODE/backendbench_tests
HUGGINGFACE_REPO = "GPUMODE/backendbench_tests"
TORCHBENCH_SUITE_HF_COMMIT = "ca7b1361b162d1499cb22ea4ad589dae506ead5d"
TORCHBENCH_SUITE_FILE = "backend_bench_problems.parquet"


def _args_size(args):
    """Calculate the size of arguments in bytes."""

    size = 0
    for arg in args:
        if isinstance(arg, torch.Tensor):
            size += arg.numel() * arg.element_size()
        elif isinstance(arg, (tuple, list)):
            size += _args_size(arg)
    return size


def _parse_trace_file(
    filename: str, filter: Optional[List[str]] = None, limit: Optional[int] = None
) -> List[Dict]:
    """
    Parse a single trace file and return a list of operation dictionaries.

    Args:
        filename: Path to trace file
        filter: Optional list of operation name filters
    """
    op_inputs = []
    op = None
    num_ops = 0

    with open(filename, "r") as f:
        lines = list(f)
        print(f"parsing {len(lines)} lines from {filename}")
        iterator = tqdm(lines, desc=f"Parsing {Path(filename).name}")
        for line in iterator:
            if m := re.match("Operator: (.*)", line):
                num_ops += 1
                if limit:
                    if num_ops > limit:
                        break
                op = m.group(1)
                # this is due to a version skew error of the pytorch version we're
                # using for developing BackendBench and what was used in tritonbench where
                # SymInt didn't exist.
                # @todo: see if we can remove this before releasing
                if op == "aten.sum.SymInt":
                    op = "aten.sum.dim_IntList"
            if m := re.match("cnt: \\d+, (.*)", line):
                assert op is not None
                args_str = m.group(1)
                cnt = int(m.group(0).split(",")[0].split(":")[1])

                if filter is None or any(f in op for f in filter):
                    is_synthetic = cnt == 0

                    op_inputs.append(
                        {
                            "uuid": hashlib.sha256(args_str.encode() + op.encode()).hexdigest(),
                            "op_name": op,
                            "args": args_str,
                            "count": cnt,
                            "is_synthetic": is_synthetic,
                        }
                    )
    return op_inputs


def _parse_trace_stream(
    stream,
    filter: Optional[List[str]] = None,
    desc: str = "Parsing stream",
    limit: Optional[int] = None,
    model_mapping: Optional[Dict] = None,
) -> List[Dict]:
    """
    Parse trace data from a text stream (e.g., from requests.Response.iter_lines()).

    Args:
        stream: Iterable of lines (strings or bytes)
        filter: Optional list of operation name filters
        desc: Description for progress bar
    """
    op_inputs = []
    op = None
    num_ops = 0
    args_to_model = {}

    iterator = tqdm(stream, desc=desc, total=len(stream))

    for line in iterator:
        # Handle bytes from response stream
        if isinstance(line, bytes):
            line = line.decode("utf-8")

        if m := re.match("Operator: (.*)", line):
            num_ops += 1
            if limit:
                if num_ops > limit:
                    break
            op = m.group(1)
            args_to_model = model_mapping[op]
            if op == "aten.sum.SymInt":
                op = "aten.sum.dim_IntList"
        if m := re.match("cnt: \\d+, (.*)", line):
            assert op is not None
            args_str = m.group(1)
            in_models = args_to_model.get(args_str, [])
            in_models_count = len(in_models)
            cnt = int(m.group(0).split(",")[0].split(":")[1])

            if filter is None or any(f in op for f in filter):
                is_synthetic = cnt == 0

                op_inputs.append(
                    {
                        "uuid": hashlib.sha256(args_str.encode() + op.encode()).hexdigest(),
                        "op_name": op,
                        "args": args_str,
                        "count": cnt,
                        "is_synthetic": is_synthetic,
                        "in_models": in_models,
                        "in_models_count": in_models_count,
                    }
                )
    return op_inputs


def _detect_format(source: Union[str, Path, None]) -> str:
    """Detect format based on source type and extension."""
    if source is None:
        return "parquet"

    if not isinstance(source, (str, Path)):
        raise ValueError(f"Unsupported source type: {type(source)}, should be str, Path, or None")

    source_str = str(source)
    if source_str.endswith(".parquet"):
        return "parquet"
    elif source_str.endswith(".txt"):
        return "trace"
    else:
        raise ValueError(
            f"Cannot auto-detect format for source: {source}. Please specify format explicitly."
        )


def load_ops_from_source(
    source: Union[str, Path, None],
    format: str = "auto",
    filter: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Load operation data from various sources and formats.

    Args:
        source: File path or URL (only trace) or None. If None, use huggingface dataset for parquet mode (default).
        format: "trace", "parquet", or "auto" (detect from file extension)
        filter: Optional list of operation name filters

    Returns:
        List of dictionaries with detailed operation info

    Auto-detection behavior:
        - None → parquet format test set from huggingface (default)
        - *.parquet → parquet format
        - *.txt → trace format
        - http*.txt → trace format
        - Other extensions → error (must specify format explicitly)
    """
    # Format detection/validation
    if format == "auto":
        format = _detect_format(source)
    elif format not in ("parquet", "trace"):
        raise ValueError(f"Unsupported format: {format}")

    # Dispatch to appropriate loader
    loaders = {"parquet": _load_from_parquet, "trace": _load_from_trace}

    return loaders[format](source, filter)


def _load_from_parquet(
    source: Optional[Union[str, Path]] = None, filter: Optional[List[str]] = None
):
    """
    Load operations from parquet file or URL.

    Args:
        source: Local file path or None. If None, use huggingface dataset (default).
        filter: Optional list of strings to filter operation names

    Returns:
        List of dictionaries containing the data
    """

    if source is None:
        # read parquet file from huggingface
        table = load_dataset(
            HUGGINGFACE_REPO,
            data_files=TORCHBENCH_SUITE_FILE,
            revision=TORCHBENCH_SUITE_HF_COMMIT,
        )["train"]
    else:
        # read parquet file directly
        table = pq.read_table(source)

    df = table.to_pandas()
    # Apply filter if provided
    if filter:
        mask = df["op_name"].apply(lambda op: any(f in op for f in filter))
        df = df[mask]

    return df.to_dict("records")


def op_list_to_benchmark_dict(ops_list: List[Dict]) -> Dict[str, List[str]]:
    """
    Convert a list of operation dictionaries to a dictionary format which can be used for benchmarking.

    Args:
        ops_list: List of dicts with 'op_name' and 'args' keys

    Returns:
        Dictionary mapping op_name to list of args strings
    """
    result = {}
    for op_data in ops_list:
        if not op_data["included_in_benchmark"]:
            continue
        op_name = op_data["op_name"]
        args = op_data["args"]
        if op_name not in result:
            result[op_name] = []
        result[op_name].append(args)
    return result


def _load_from_trace(
    source: Union[str, Path],
    filter: Optional[List[str]],
    limit: Optional[int] = None,
    model_mapping: Optional[Dict] = None,
) -> List[Dict]:
    """Load operations from trace file(s) and return list of dicts."""
    op_inputs = []
    assert model_mapping is not None

    # Handle URLs - stream directly without saving to disk
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
        logging.info(f"Downloading trace from {source}")
        with requests.get(source) as response:
            response.raise_for_status()

            # Download entire content
            content = response.text

            # Create an iterator from the lines for the progress bar
            lines = content.splitlines()

            # Now parse with accurate progress (tqdm will know total lines)
            op_inputs = _parse_trace_stream(
                lines, filter, "Parsing", limit=limit, model_mapping=model_mapping
            )

    # Handle single files
    else:
        op_inputs = _parse_trace_file(source, filter, limit=limit)

    return op_inputs
