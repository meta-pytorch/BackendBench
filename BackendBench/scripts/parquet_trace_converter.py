# utility functions to convert parquet and trace files back and forth

import pyarrow.parquet as pq
import pyarrow.csv as csv
import pyarrow as pa
from BackendBench.torchbench_suite import DEFAULT_HUGGINGFACE_URL, _args_size
from BackendBench.utils import deserialize_args
import os
import requests
import tempfile
from pathlib import Path
import hashlib
import re
from tqdm import tqdm

"""
For the dataset release we generally would want to versions
1. A production version which has what you would want to run a benchmark with an llm
2. A "dev" version. This version is much more verbose, has more information on each test, includes tests/ops we decided to axe (and why they were axed), and possibly some runtime numbers

The point of 1 is for something to have folks able to benchmark their agents against. Therefore, there is a high quality bar for inclusion
At the end of the day we still need solutions to be general for inclusion in pytorch, therefore, the mroe verbose dev version is useful in this case. It also allows us to record information on the ops and decisions as well

Columns for the production version:
- uuid (int) (hash of op + args)
- op_name (string)
- args (string)
- arg size (float)(in MB)
- count (int) (number of times this op + set of args was called in real models)
- is_synthetic (boolean) (did we generate this op or is it from a real model)


Columns for the dev version:
All columns in the production version, plus:
- include_in_prod (boolean)
- why_excluded (string) (empty if included)
- runtime_ms (float) (timings on H100 gpu)
- runnable (bool) (does this op + test work) [we may remove this column later after we solve for special ops]
- in_models (string) (which models did we include this op in)
"""

def _parse_trace(filename):
    
    # given a trace file it returns a list of dicts which include
    # uuid, op_name, args, arg_size, count

    op_inputs = []

    with open(filename, "r") as f:
        for line in tqdm(f, desc="Parsing trace file"):
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
                op_inputs.append({
                    "uuid": hashlib.sha256(args_str.encode() + op.encode()).hexdigest(),
                    "op_name": op,
                    "args": args_str,
                    "arg_size": size,
                    "count": cnt,
                    "is_synthetic": is_synthetic,
                })
    return op_inputs

def convert_trace_to_parquets(trace_file, prod_parquet_file=None, dev_parquet_file=None):
    """
    Convert a trace file to a parquet file
    """

    ops = []

    # Check if filename is a URL
    if isinstance(trace_file, str) and (
        trace_file.startswith("http://") or trace_file.startswith("https://")
    ):
        with (
            tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file,
            requests.get(trace_file) as response,
        ):
            response.raise_for_status()
            tmp_file.write(response.text)
            tmp_file.flush()
            ops.extend(_parse_trace(tmp_file.name))
            Path(tmp_file.name).unlink(missing_ok=True)
    elif Path(trace_file).is_dir():
        for file_path in Path(trace_file).glob("**/*.txt"):
            ops.extend(_parse_trace(str(file_path)))
    else:
        ops.extend(_parse_trace(trace_file))

    # create dict for dev version
    print(ops)

    
def convert_parquet_to_trace(parquet_file, trace_file):
    """
    Convert a parquet file to a trace file
    """
    pass

if __name__ == "__main__":
    file_path = DEFAULT_HUGGINGFACE_URL
    convert_trace_to_parquets(file_path, "prod.parquet", "dev.parquet")