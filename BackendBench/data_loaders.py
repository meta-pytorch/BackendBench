"""
Shared data loading utilities for reading trace and parquet files.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
import pyarrow.parquet as pq
import torch
from BackendBench.utils import deserialize_args
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


def _parse_trace_file(filename: str, filter: Optional[List[str]] = None) -> List[Dict]:
    """
    Parse a single trace file and return a list of operation dictionaries.

    Args:
        filename: Path to trace file
        filter: Optional list of operation name filters
    """
    op_inputs = []
    op = None

    with open(filename, "r") as f:
        lines = list(f)
        iterator = tqdm(lines, desc=f"Parsing {Path(filename).name}")
        for line in iterator:
            if m := re.match("Operator: (.*)", line):
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
                    size = size / (1024 * 1024)  # Convert to MB
                    is_synthetic = cnt == 0

                    op_inputs.append(
                        {
                            "uuid": hashlib.sha256(args_str.encode() + op.encode()).hexdigest(),
                            "op_name": op,
                            "args": args_str,
                            "arg_size": size,
                            "count": cnt,
                            "is_synthetic": is_synthetic,
                        }
                    )
    return op_inputs


def _parse_trace_stream(
    stream, filter: Optional[List[str]] = None, desc: str = "Parsing stream"
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

    iterator = tqdm(stream, desc=desc)

    for line in iterator:
        # Handle bytes from response stream
        if isinstance(line, bytes):
            line = line.decode("utf-8")

        if m := re.match("Operator: (.*)", line):
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
                size = size / (1024 * 1024)  # Convert to MB
                is_synthetic = cnt == 0

                op_inputs.append(
                    {
                        "uuid": hashlib.sha256(args_str.encode() + op.encode()).hexdigest(),
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
        source: File path, URL, or directory
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
        - directory_path/ → trace format (scans for .txt files)
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
                # Local path - check if it's a directory
                if Path(source).is_dir():
                    format = "trace"  # Directory scan for .txt files
                else:
                    format = "trace"  # Default to trace
        else:
            format = "trace"

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


def ops_list_to_dict(ops_list: List[Dict]) -> Dict[str, List[str]]:
    """
    Convert a list of operation dictionaries to a dictionary format.

    Args:
        ops_list: List of dicts with 'op_name' and 'args' keys

    Returns:
        Dictionary mapping op_name to list of args strings
    """
    result = {}
    for op_data in ops_list:
        op_name = op_data["op_name"]
        args = op_data["args"]
        if op_name not in result:
            result[op_name] = []
        result[op_name].append(args)
    return result


def _load_from_trace(source: Union[str, Path], filter: Optional[List[str]]) -> List[Dict]:
    """Load operations from trace file(s) and return list of dicts."""
    op_inputs = []

    # Handle URLs - stream directly without saving to disk
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
        with requests.get(source, stream=True) as response:
            response.raise_for_status()
            desc = f"Parsing {source}"
            op_inputs = _parse_trace_stream(response.iter_lines(), filter, desc)

    # Handle directories
    elif Path(source).is_dir():
        for file_path in Path(source).glob("**/*.txt"):
            op_inputs.extend(_parse_trace_file(str(file_path), filter))

    # Handle single files
    else:
        op_inputs = _parse_trace_file(source, filter)

    return op_inputs
