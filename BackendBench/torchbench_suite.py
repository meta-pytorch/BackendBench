"""
Load aten inputs from serialized txt files.
"""

import re
import tempfile
from collections import defaultdict
from pathlib import Path

import requests
import torch
from BackendBench.utils import deserialize_args

# the schema for this dataset is the one defined in tritonbench traces.
# ie. https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/data/input_configs/hf_train/AlbertForMaskedLM_training.txt
DEFAULT_HUGGINGFACE_URL = "https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/tritonbench_op_trace.txt"


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
            args, kwargs = deserialize_args(inp)
            size = _args_size(args) + _args_size(list(kwargs.values()))
            inputs_and_sizes.append((size, inp))
        ret = [x[1] for x in sorted(inputs_and_sizes, reverse=True)]
        return ret if self.topn is None else ret[: self.topn]

    @property
    def correctness_tests(self):
        for inp in self.tests():
            args, kwargs = deserialize_args(inp)
            yield TorchBenchTest(*args, **kwargs)

    @property
    def performance_tests(self):
        for inp in self.tests():
            args, kwargs = deserialize_args(inp)
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
    def __init__(self, name, filename=None, filter=None, topn=None):
        self.name = name
        self.topn = topn
        self.optests = defaultdict(list)

        # Use default URL if no filename provided
        if filename is None:
            filename = DEFAULT_HUGGINGFACE_URL

        # Check if filename is a URL
        if isinstance(filename, str) and (
            filename.startswith("http://") or filename.startswith("https://")
        ):
            with (
                tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file,
                requests.get(filename) as response,
            ):
                response.raise_for_status()
                tmp_file.write(response.text)
                tmp_file.flush()
                _parse_inputs(tmp_file.name, filter, self.optests)
                Path(tmp_file.name).unlink(missing_ok=True)
        elif Path(filename).is_dir():
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
                    "_cudnn_rnn_backward.default",  # RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM
                    "_fft_c2c.default",  # cuFFT only supports dimensions whose sizes are powers of two when computing in half precision
                ]
            ):
                # TODO: indexing ops need valid indices
                continue
            yield TorchBenchOpTest(op, inputs, self.topn)
