

"""
Load aten inputs from serialized txt files.
"""

import re
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.testing import make_tensor


dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

dtype_abbrs_parsing = {value: key for key, value in dtype_abbrs.items()}

_FLOATING_TYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _deserialize_tensor(size, dtype, stride=None, device="cuda"):
    if stride is not None:
        out = torch.empty_strided(size, stride, dtype=dtype, device=device)
    else:
        out = torch.empty(size, dtype=dtype, device=device)
    if dtype in _FLOATING_TYPES:
        return out.copy_(make_tensor(size, dtype=dtype, device=device, low=0, high=1))
    return out.copy_(make_tensor(size, dtype=dtype, device=device))


def _deserialize_args(inps):
    inps = inps.strip().strip("'")
    global_vals = {
        "T": _deserialize_tensor,
        "th": torch,
        "inf": math.inf,
        "torch": torch,
        **dtype_abbrs_parsing,
    }
    # f strings introduce quotations we dont want
    for key in dtype_abbrs_parsing:
        inps = inps.replace(f"'{key}'", key)
    return eval(inps.strip().strip("'").strip('"'), global_vals)


class TorchBenchTest:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TorchBenchOpTest:
    def __init__(self, op, inputs):
        self.op = eval(f"torch.ops.{op}")
        self.inputs = inputs

    @property
    def correctness_tests(self):
        for inp in self.inputs:
            args, kwargs = _deserialize_args(inp)
            yield TorchBenchTest(*args, **kwargs)

    @property
    def performance_tests(self):
        for inp in self.inputs:
            args, kwargs = _deserialize_args(inp)
            yield TorchBenchTest(*args, **kwargs)


def _parse_inputs(filename, filter, op_inputs):
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


class TorchBenchTestSuite:
    def __init__(self, name, filename, filter=None):
        self.name = name
        self.optests = defaultdict(list)
        if Path(filename).is_dir():
            for file_path in Path(filename).glob("**/*.txt"):
                _parse_inputs(str(file_path), filter, self.optests)
        else:
            _parse_inputs(filename, filter, self.optests)
        # Deduplicate the strings in self.optests
        for op in self.optests:
            self.optests[op] = list(set(self.optests[op]))

    def __iter__(self):
        for op, inputs in self.optests.items():
            if any(
                s in op for s in ["embedding", "scatter", "gather", "index", "nll_loss"]
            ):
                # TODO: indexing ops need valid indices
                continue
            yield TorchBenchOpTest(op, inputs)
