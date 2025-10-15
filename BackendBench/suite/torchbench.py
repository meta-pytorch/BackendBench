# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite that runs real-world PyTorch operation traces from serialized data files.

Data Source:
- Dataset: https://huggingface.co/datasets/GPUMODE/backendbench_tests
- Configuration: Set in data_loaders.py:
  - HUGGINGFACE_REPO: HF repository name
  - TORCHBENCH_SUITE_FILE: Specific file name in the repo
  - TORCHBENCH_SUITE_HF_COMMIT: Git commit hash for reproducibility

Updating the Test Set:
1. Choose a test file from https://huggingface.co/datasets/GPUMODE/backendbench_tests (it will likely be the same)
2. Update TORCHBENCH_SUITE_FILE in data_loaders.py with the file name (it will likely be the same)
3. Get the current commit hash:
   python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('GPUMODE/backendbench_tests', revision='main').sha)"
4. Update TORCHBENCH_SUITE_HF_COMMIT in data_loaders.py with the hash

Creating New Test Sets:
Use scripts/parquet_to_trace.py to generate and upload new datasets to HuggingFace.
"""

import torch  # noqa: F401

from BackendBench.data_loaders import (
    _args_size,
    load_ops_from_source,
    op_list_to_benchmark_dict,
)
from BackendBench.op_categories import UNSUPPORTED_OPERATORS
from BackendBench.utils import deserialize_args


class TorchBenchTest:
    def __init__(self, *args, test_backwards=False, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.test_backwards = test_backwards


class TorchBenchOpTest:
    def __init__(self, op, inputs, topn, check_backwards=False):
        self.op = eval(f"torch.ops.{op}")
        self.inputs = inputs
        self.topn = topn
        self._check_backwards = check_backwards

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
        from BackendBench.backwards_utils import should_check_backwards_for_op

        # Determine if this op should check backwards
        test_backwards = should_check_backwards_for_op(self.op.__name__, self._check_backwards)

        for inp in self.tests():
            args, kwargs = deserialize_args(inp)
            yield TorchBenchTest(*args, test_backwards=test_backwards, **kwargs)

    @property
    def performance_tests(self):
        for inp in self.tests():
            args, kwargs = deserialize_args(inp)
            yield TorchBenchTest(*args, **kwargs)


class TorchBenchTestSuite:
    def __init__(
        self,
        name,
        filename=None,
        filter=None,
        topn=None,
        check_overhead_dominated_ops=False,
        check_backwards=False,
    ):
        self.name = name
        self.topn = topn
        self.check_backwards = check_backwards
        # Load operations using the shared data loader
        ops_list = load_ops_from_source(
            source=filename,
            format="auto",  # Auto-detect based on file extension
            filter=filter,
        )
        if check_overhead_dominated_ops:
            # Only include ops which are overhead dominated (this is useful as a performance canary)
            ops_list = [op for op in ops_list if op.get("is_overhead_dominated_op", False)]

        # Convert to dictionary format using utility function
        self.optests = op_list_to_benchmark_dict(ops_list)

        # Deduplicate the strings in self.optests
        for op in self.optests:
            self.optests[op] = list(set(self.optests[op]))

    def __iter__(self):
        for op, inputs in self.optests.items():
            if any(s in op for s in UNSUPPORTED_OPERATORS):
                continue
            yield TorchBenchOpTest(op, inputs, self.topn, self.check_backwards)
