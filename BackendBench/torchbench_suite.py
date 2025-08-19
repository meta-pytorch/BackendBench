# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Load aten inputs from serialized txt files and parquet files.
"""

import torch  # noqa: F401
from BackendBench.data_loaders import (
    _args_size,
    load_ops_from_source,
    op_list_to_benchmark_dict,
)
from BackendBench.scripts.dataset_filters import SKIP_OPERATORS
from BackendBench.utils import deserialize_args

# for details on the dataset read this:
# https://huggingface.co/datasets/GPUMODE/huggingface_op_trace
DEFAULT_HUGGINGFACE_URL = "https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/backend_bench_problems.parquet"


class TorchBenchTest:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


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


class TorchBenchTestSuite:
    def __init__(self, name, filename=None, filter=None, topn=None):
        self.name = name
        self.topn = topn

        # Use default URL if no filename provided
        if filename is None:
            filename = DEFAULT_HUGGINGFACE_URL

        # Load operations using the shared data loader
        ops_list = load_ops_from_source(
            source=filename,
            format="auto",  # Auto-detect based on file extension
            filter=filter,
        )

        # Convert to dictionary format using utility function
        self.optests = op_list_to_benchmark_dict(ops_list)

        # Deduplicate the strings in self.optests
        for op in self.optests:
            self.optests[op] = list(set(self.optests[op]))

    def __iter__(self):
        for op, inputs in self.optests.items():
            if any(s in op for s in SKIP_OPERATORS):
                continue
            yield TorchBenchOpTest(op, inputs, self.topn)
