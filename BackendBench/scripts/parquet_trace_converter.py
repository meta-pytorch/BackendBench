# utility functions to convert parquet and trace files back and forth

import pyarrow.parquet as pq
import pyarrow as pa
from BackendBench.torchbench_suite import DEFAULT_HUGGINGFACE_URL
from BackendBench.data_loaders import _args_size
from BackendBench.scripts.dataset_filters import (
    _apply_skip_ops_filter,
    _apply_non_interesting_ops_filter,
)
from BackendBench.utils import deserialize_args
import os
import logging
import click
import re
import hashlib
from tqdm import tqdm
import tempfile
import requests
from pathlib import Path
from typing import List, Dict

"""
For the dataset release we generally would want to versions
1. A production version which has what you would want to run a benchmark with an llm
2. A "dev" version. This version is much more verbose, has more information on each test, includes tests/ops we decided to axe (and why they were axed), and possibly some runtime numbers

The point of 1 is for something to have folks able to benchmark their agents against. Therefore, there is a high quality bar for inclusion
At the end of the day we still need solutions to be general for inclusion in pytorch, therefore, the more verbose dev version is useful in this case. It also allows us to record information on the ops and decisions as well

Columns for the production version:
- uuid (int) (hash of op + args)
- op_name (string)
- args (string)
- arg size (float)(in MB)
- count (int) (number of times this op + set of args was called in real models)
- is_synthetic (boolean) (did we generate this op or is it from a real model)


Columns for the dev version:
All columns in the production version, plus:
- included_in_benchmark (boolean)
- why_excluded (list of strings) (empty if included)
- runtime_ms (float) (timings on H100 gpu)
- runnable (bool) (does this op + test work) [we may remove this column later after we solve for special ops]
- in_models (string) (which models did we include this op in) [@TODO add this]
"""

logger = logging.getLogger(__name__)


def setup_logging(log_level):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s][%(levelname)s][%(filename)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_trace_file(filename: str) -> List[Dict]:
    """
    Parse a single trace file and return a list of operation dictionaries.

    Returns list of dicts with keys: uuid, op_name, args, arg_size, count, is_synthetic
    """
    op_inputs = []
    op = None

    with open(filename, "r") as f:
        for line in tqdm(f, desc=f"Parsing {Path(filename).name}"):
            if m := re.match("Operator: (.*)", line):
                op = m.group(1)
                if op == "aten.sum.SymInt":
                    op = "aten.sum.dim_IntList"
            if m := re.match("cnt: \\d+, (.*)", line):
                assert op is not None
                args_str = m.group(1)
                # extract cnt value from group 0
                cnt = int(m.group(0).split(",")[0].split(":")[1])
                args, kwargs = deserialize_args(args_str)
                size = _args_size(args) + _args_size(list(kwargs.values()))
                # convert size to MB from bytes
                size = size / (1024 * 1024)
                # if cnt is 0 then it is synthetic
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


def _load_trace_for_parquet_conversion(source: str) -> List[Dict]:
    """
    Load operations from trace file(s) with detailed metadata for parquet conversion.
    """
    ops = []

    # Handle URLs
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
        with (
            tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file,
            requests.get(source) as response,
        ):
            response.raise_for_status()
            tmp_file.write(response.text)
            tmp_file.flush()
            ops.extend(_parse_trace_file(tmp_file.name))
            Path(tmp_file.name).unlink(missing_ok=True)

    # Handle directories
    elif Path(source).is_dir():
        for file_path in Path(source).glob("**/*.txt"):
            ops.extend(_parse_trace_file(str(file_path)))

    # Handle single files
    else:
        ops.extend(_parse_trace_file(source))

    return ops


def convert_trace_to_parquets(trace_file, prod_parquet_file=None, dev_parquet_file=None):
    """
    Convert a trace file to a parquet file
    """

    # Load operations using local trace parsing function
    ops = _load_trace_for_parquet_conversion(trace_file)

    # Add additional metadata fields required for the parquet format
    for op in ops:
        op["uuid"] = hashlib.sha256(op["args"].encode() + op["op_name"].encode()).hexdigest()
        op["included_in_benchmark"] = True
        op["why_excluded"] = []
        op["runtime_ms"] = ""
        op["runnable"] = True

    # apply filters
    ops = _apply_skip_ops_filter(ops)
    ops = _apply_non_interesting_ops_filter(ops)

    # create production version (copy ops to avoid modifying original)
    prod_ops = []
    for op in ops:
        if op["included_in_benchmark"]:
            # Create production version without metadata fields
            prod_op = {
                "uuid": op["uuid"],
                "op_name": op["op_name"],
                "args": op["args"],
                "arg_size": op["arg_size"],
                "count": op["count"],
                "is_synthetic": op["is_synthetic"],
            }
            prod_ops.append(prod_op)

    # Create parquet tables with proper headers
    dev_table = pa.Table.from_pylist(ops)
    prod_table = pa.Table.from_pylist(prod_ops)

    # Write parquet files
    pq.write_table(dev_table, dev_parquet_file)
    pq.write_table(prod_table, prod_parquet_file)

    logger.info(f"Wrote {len(prod_ops)} ops and inputs to {prod_parquet_file}")
    logger.info(f"Wrote {len(ops)} ops and inputs to {dev_parquet_file}")

    # Log column information for verification
    logger.debug(f"Production parquet columns: {prod_table.column_names}")
    logger.debug(f"Dev parquet columns: {dev_table.column_names}")


def convert_parquet_to_trace(parquet_file, trace_file):
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

    # write to trace file
    with open(trace_file, "w") as f:
        for op, args in op_inputs.items():
            f.write(f"Operator: {op}\n")
            for arg in args:
                f.write(f"{arg}\n")
    total_args = sum(len(op_inputs[op]) for op in op_inputs)
    logging.info(f"Wrote {total_args} ops and inputs to {trace_file}")


def _validate_parquet_name(parquet_name: str, is_input: bool = False) -> str:
    """Validate parquet filename. URLs allowed only for inputs."""
    # URLs are allowed only if this is an input file
    if parquet_name.startswith(("http://", "https://")):
        if is_input:
            return parquet_name  # URL allowed for input
        else:
            raise click.BadParameter("Output parquet file cannot be a URL")

    if not parquet_name.endswith(".parquet"):
        raise click.BadParameter("Parquet file must end with .parquet suffix")

    # Ensure local files are in datasets directory
    if not parquet_name.startswith("datasets/"):
        parquet_name = f"datasets/{parquet_name}"

    return parquet_name


def _validate_trace_file(trace_file: str, is_input: bool = True) -> str:
    """Validate trace file. URLs allowed only for inputs."""
    # URLs are allowed only if this is an input file
    if trace_file.startswith(("http://", "https://")):
        if is_input:
            return trace_file  # URL allowed for input
        else:
            raise click.BadParameter("Output trace file cannot be a URL")

    # For local files, check extension
    if not (trace_file.endswith(".txt") or Path(trace_file).is_dir()):
        raise click.BadParameter("Local trace file must end with .txt or be a directory")

    if Path(trace_file).is_dir() and not is_input:
        raise click.BadParameter("Output trace file cannot be a directory")

    return trace_file


def _generate_dev_parquet_name(parquet_name: str) -> str:
    """Generate dev parquet name by appending -dev before .parquet suffix."""
    if parquet_name.endswith(".parquet"):
        base_name = parquet_name[:-8]  # Remove .parquet
        return f"{base_name}-dev.parquet"
    else:
        return f"{parquet_name}-dev"


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
    default=DEFAULT_HUGGINGFACE_URL,
    type=str,
    help="Input trace file: URL (for downloads), local .txt file, or directory. Output trace files cannot be URLs",
)
@click.option(
    "--parquet-name",
    default="backend_bench_problems-dev.parquet",
    type=str,
    help="Parquet filename: URL allowed as input in parquet-to-trace mode, local files in datasets/. Dev version auto-generated as filename-dev.parquet.",
)
def main(log_level, mode, trace_file, parquet_name):
    """Convert trace files to parquet format or vice versa."""
    setup_logging(log_level)

    # Create datasets directory
    os.makedirs("datasets", exist_ok=True)

    if mode == "trace-to-parquet":
        # Validate inputs/outputs
        trace_file = _validate_trace_file(trace_file, is_input=True)  # Input: URLs allowed
        parquet_name = _validate_parquet_name(
            parquet_name, is_input=False
        )  # Output: URLs not allowed

        # Generate dev version name
        dev_parquet_name = _generate_dev_parquet_name(parquet_name)

        logger.info(f"Converting trace file {trace_file} to parquet files")
        logger.info(f"Production parquet: {parquet_name}")
        logger.info(f"Dev parquet: {dev_parquet_name}")

        convert_trace_to_parquets(trace_file, parquet_name, dev_parquet_name)
        logger.info("Conversion completed successfully")

    elif mode == "parquet-to-trace":
        # Validate parquet input (URLs allowed for input in this mode)
        parquet_input = _validate_parquet_name(parquet_name, is_input=True)  # Input: URLs allowed
        # Validate trace output (URLs not allowed for output)
        trace_output = _validate_trace_file(trace_file, is_input=False)  # Output: URLs not allowed

        logger.info(f"Converting parquet file {parquet_input} to trace file {trace_output}")
        convert_parquet_to_trace(parquet_input, trace_output)
        logger.info("Conversion completed successfully")


if __name__ == "__main__":
    main()
