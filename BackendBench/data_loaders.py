# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared data loading utilities for reading trace and parquet files.
"""

import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pyarrow.parquet as pq

import requests
import torch
from BackendBench.utils import cleanup_memory_and_gpu, deserialize_args
from tqdm import tqdm


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
                    args, kwargs = deserialize_args(args_str)
                    size = _args_size(args) + _args_size(list(kwargs.values()))
                    size = size / (1024 * 1024)  # Convert to MB
                    del args, kwargs
                    cleanup_memory_and_gpu()
                    is_synthetic = cnt == 0

                    op_inputs.append(
                        {
                            "uuid": hashlib.sha256(
                                args_str.encode() + op.encode()
                            ).hexdigest(),
                            "op_name": op,
                            "args": args_str,
                            "arg_size": size,
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
            if op == "aten.sum.SymInt":
                op = "aten.sum.dim_IntList"
        if m := re.match("cnt: \\d+, (.*)", line):
            assert op is not None
            args_str = m.group(1)
            cnt = int(m.group(0).split(",")[0].split(":")[1])

            if filter is None or any(f in op for f in filter):
                args, kwargs = deserialize_args(args_str)
                size = _args_size(args) + _args_size(list(kwargs.values()))
                del args, kwargs
                cleanup_memory_and_gpu()
                size = size / (1024 * 1024)  # Convert to MB
                is_synthetic = cnt == 0

                op_inputs.append(
                    {
                        "uuid": hashlib.sha256(
                            args_str.encode() + op.encode()
                        ).hexdigest(),
                        "op_name": op,
                        "args": args_str,
                        "arg_size": size,
                        "count": cnt,
                        "is_synthetic": is_synthetic,
                    }
                )
    return op_inputs


def load_ops_from_source(
    source: Union[str, Path],
    format: str = "auto",
    filter: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Load operation data from various sources and formats.

    Args:
        source: File path or URL
        format: "trace", "parquet", or "auto" (detect from file extension)
        filter: Optional list of operation name filters

    Returns:
        List of dictionaries with detailed operation info

    Auto-detection behavior:
        - https://domain.com/data.parquet → parquet format
        - https://domain.com/data.txt → trace format
        - https://domain.com/data → trace format (fallback)
        - local_file.parquet → parquet format
        - local_file.txt → trace format
    """

    # Auto-detect format if not specified
    if format == "auto":
        if isinstance(source, str):
            # Check file extension first (works for both local files and URLs)
            if source.endswith(".parquet"):
                format = "parquet"
            elif source.endswith(".txt"):
                format = "trace"
            elif source.startswith(("http://", "https://")):
                # Remote URL without recognizable extension - default to trace
                format = "trace"
            else:
                raise ValueError(f"Unsupported source: {source}")
        else:
            raise ValueError(f"Unsupported source: {source}")

    if format == "parquet":
        return _load_from_parquet(source, filter)
    elif format == "trace":
        # Always load full data - consumers can extract what they need
        return _load_from_trace(source, filter)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_from_parquet(source: Union[str, Path], filter: Optional[List[str]]):
    """Load operations from parquet file."""
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
    source: Union[str, Path], filter: Optional[List[str]], limit: Optional[int] = None
) -> List[Dict]:
    """Load operations from trace file(s) and return list of dicts."""
    op_inputs = []

    # Handle URLs - stream directly without saving to disk
    if isinstance(source, str) and (
        source.startswith("http://") or source.startswith("https://")
    ):
        logging.info(f"Downloading trace from {source}")
        with requests.get(source) as response:
            response.raise_for_status()

            # Download entire content
            content = response.text

            # Create an iterator from the lines for the progress bar
            lines = content.splitlines()

            # Now parse with accurate progress (tqdm will know total lines)
            op_inputs = _parse_trace_stream(lines, filter, "Parsing", limit=limit)

    # Handle single files
    else:
        op_inputs = _parse_trace_file(source, filter, limit=limit)

    return op_inputs
