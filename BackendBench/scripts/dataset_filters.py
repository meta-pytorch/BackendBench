# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import torch

import tqdm
from BackendBench.utils import cleanup_memory_and_gpu, deserialize_args
from triton.testing import do_bench

# Operators to skip for indexing ops that need valid indices
SKIP_OPERATORS = [
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

# We get this threshhold from the analysis here
# https://github.com/meta-pytorch/BackendBench/issues/108
RELATIVE_RUNTIME_THRESHOLD = 1.3
UNTESTABLE_OPERATORS = [
    "empty_like",  # We can check using metadata
    "new_empty",  # We can check using metadata
    "new_empty_strided",  # We can check using metadata
    "bernoulli",  # We can write a custom test to verify this one (albeit not the randomness)
]


def apply_skip_ops_filter(ops):
    for op in tqdm.tqdm(ops, desc="Filtering ops by skip and synthetic ops"):
        if any(skip_op in op["op_name"] for skip_op in SKIP_OPERATORS):
            op["included_in_benchmark"] = False
            op["why_excluded"].append("We cannot run this op on backendbench yet")
            op["runnable"] = False

        if any(skip_op in op["op_name"] for skip_op in UNTESTABLE_OPERATORS):
            op["included_in_benchmark"] = False
            op["why_excluded"].append(
                "BackendBench does not support correctness testing for this op yet"
            )

        if op["is_synthetic"]:
            op["included_in_benchmark"] = False
            op["why_excluded"].append(
                "Synthetic ops are not supported in the official benchmark yet"
            )
            op["runnable"] = False
    return ops


def apply_runtime_filter(ops):
    def _overhead_benchmark():
        return torch.randn(1, device="cuda")

    runtime_threshold_ms = do_bench(_overhead_benchmark, warmup=25, rep=100)

    for op in tqdm.tqdm(ops, desc="Filtering ops by runtime"):
        if op["runnable"]:
            args, kwargs = deserialize_args(op["args"])
            try:
                op_name = op["op_name"]
                op_func = eval(f"torch.ops.{op_name}")
                ms = do_bench(lambda: op_func(*args, **kwargs), warmup=25, rep=100)
                del args, kwargs
                cleanup_memory_and_gpu()
            except Exception as e:
                # if we can't run the op, we cannot expect others to run it either
                op["why_excluded"].append(f"Failed to run: {e}")
                op["runnable"] = False
                op["included_in_benchmark"] = False
                del args, kwargs
                cleanup_memory_and_gpu()
                continue
            op["runtime_ms"] = ms
            relative_runtime = ms / runtime_threshold_ms
            op["relative_runtime_to_kernel_launch"] = relative_runtime
            if relative_runtime < RELATIVE_RUNTIME_THRESHOLD:
                op["included_in_benchmark"] = False
                op["performance_canary"] = True
    return ops
