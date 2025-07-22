"""
Load aten inputs from serialized txt files.
"""

import math
import re
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
    kwargs = {}
    if dtype in _FLOATING_TYPES:
        kwargs.update({"low": 0, "high": 1})
    if stride is not None:
        extent = 1 + sum((size - 1) * stride for size, stride in zip(size, stride))
        data = make_tensor(extent, dtype=dtype, device=device, **kwargs)
        return data.as_strided(size, stride)
    return make_tensor(size, dtype=dtype, device=device, **kwargs)


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


def _args_size(args):
    size = 0
    for arg in args:
        if isinstance(arg, torch.Tensor):
            size += arg.numel() * arg.element_size()
        elif isinstance(arg, (tuple, list)):
            size += _args_size(arg)
    return size


class TorchBenchOpTest:
    def __init__(self, op, inputs, topn):
        self.op = eval(f"torch.ops.{op}")
        self.inputs = inputs
        self.topn = topn

    def tests(self):
        inputs_and_sizes = []
        for inp in self.inputs:
            args, kwargs = _deserialize_args(inp)
            size = _args_size(args) + _args_size(list(kwargs.values()))
            inputs_and_sizes.append((size, inp))
        ret = [x[1] for x in sorted(inputs_and_sizes, reverse=True)]
        return ret if self.topn is None else ret[: self.topn]

    @property
    def correctness_tests(self):
        for inp in self.tests():
            args, kwargs = _deserialize_args(inp)
            yield TorchBenchTest(*args, **kwargs)

    @property
    def performance_tests(self):
        for inp in self.tests():
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
    def __init__(self, name, filename, filter=None, topn=None):
        self.name = name
        self.topn = topn
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
                s in op
                for s in [
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
                ]
            ):
                # TODO: indexing ops need valid indices
                continue
            yield TorchBenchOpTest(op, inputs, self.topn)
