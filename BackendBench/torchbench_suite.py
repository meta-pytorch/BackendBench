"""
Load aten inputs from serialized txt files and parquet files.
"""

import torch  # noqa: F401
from BackendBench.data_loaders import _args_size, load_ops_from_source
from BackendBench.scripts.dataset_filters import SKIP_OPERATORS
from BackendBench.utils import deserialize_args

# the schema for this dataset is the one defined in tritonbench traces.
# ie. https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/data/input_configs/hf_train/AlbertForMaskedLM_training.txt
DEFAULT_HUGGINGFACE_URL = "https://huggingface.co/datasets/GPUMODE/huggingface_op_trace/resolve/main/tritonbench_op_trace.txt"


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
        # Use simple_format=True to get the defaultdict format for compatibility
        self.optests = load_ops_from_source(
            source=filename,
            format="auto",  # Auto-detect based on file extension
            filter=filter,
            simple_format=True,
        )

        # Deduplicate the strings in self.optests
        for op in self.optests:
            self.optests[op] = list(set(self.optests[op]))

    def __iter__(self):
        for op, inputs in self.optests.items():
            if any(s in op for s in SKIP_OPERATORS):
                # TODO: indexing ops need valid indices
                continue
            yield TorchBenchOpTest(op, inputs, self.topn)
