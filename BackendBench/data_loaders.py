"""
Shared data loading utilities for reading trace and parquet files.
"""

import re
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Union

import requests
import pyarrow.parquet as pq
import torch


def _args_size(args):
    """Calculate the size of arguments in bytes."""

    size = 0
    for arg in args:
        if isinstance(arg, torch.Tensor):
            size += arg.numel() * arg.element_size()
        elif isinstance(arg, (tuple, list)):
            size += _args_size(arg)
    return size


def _parse_trace_file_simple(filename: str, filter: Optional[List[str]], op_inputs: Dict) -> Dict:
    """
    Parse a single trace file for TorchBenchSuite (simpler format).

    Returns defaultdict where keys are op names and values are lists of args strings.
    """
    op = None

    with open(filename, "r") as f:
        for line in f:
            if m := re.match("Operator: (.*)", line):
                op = m.group(1)
                if op == "aten.sum.SymInt":
                    op = "aten.sum.dim_IntList"
            if m := re.match("cnt: \\d+, (.*)", line):
                assert op is not None
                args = m.group(1)
                if filter is None or any(f in op for f in filter):
                    op_inputs[op].append(args)
    return op_inputs


def load_ops_from_source(
    source: Union[str, Path],
    format: str = "auto",
    filter: Optional[List[str]] = None,
    simple_format: bool = False,
) -> Union[List[Dict], Dict]:
    """
    Load operation data from various sources and formats.

    Args:
        source: File path, URL, or directory
        format: "trace", "parquet", or "auto" (detect from file extension)
        filter: Optional list of operation name filters
        simple_format: If True, return defaultdict format for TorchBenchSuite compatibility

    Returns:
        If simple_format=True: defaultdict with op names as keys, args lists as values
        If simple_format=False: List of dictionaries with detailed operation info

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
        return _load_from_parquet(source, filter, simple_format)
    elif format == "trace":
        return _load_from_trace(source, filter, simple_format)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_from_parquet(source: Union[str, Path], filter: Optional[List[str]], simple_format: bool):
    """Load operations from parquet file."""
    table = pq.read_table(source)

    if simple_format:
        # Convert to TorchBenchSuite format
        op_inputs = defaultdict(list)
        for batch in table.to_batches():
            df = batch.to_pandas()
            for _, row in df.iterrows():
                op_name = row["op_name"]
                if filter is None or any(f in op_name for f in filter):
                    op_inputs[op_name].append(row["args"])
        return op_inputs
    else:
        # Convert to list of dicts
        df = table.to_pandas()
        return df.to_dict("records")


def _load_from_trace(source: Union[str, Path], filter: Optional[List[str]], simple_format: bool):
    """Load operations from trace file(s). Only supports simple_format=True for TorchBenchSuite."""
    if not simple_format:
        raise ValueError(
            "Detailed trace parsing has been moved to parquet_trace_converter.py. Use simple_format=True."
        )

    op_inputs = defaultdict(list)

    # Handle URLs
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
        with (
            tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file,
            requests.get(source) as response,
        ):
            response.raise_for_status()
            tmp_file.write(response.text)
            tmp_file.flush()
            _parse_trace_file_simple(tmp_file.name, filter, op_inputs)
            Path(tmp_file.name).unlink(missing_ok=True)

    # Handle directories
    elif Path(source).is_dir():
        for file_path in Path(source).glob("**/*.txt"):
            _parse_trace_file_simple(str(file_path), filter, op_inputs)

    # Handle single files
    else:
        _parse_trace_file_simple(source, filter, op_inputs)

    return op_inputs
