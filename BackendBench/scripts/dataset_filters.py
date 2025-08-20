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


def apply_skip_ops_filter(ops):
    total_ops = 0
    synthetic_ops = 0
    skip_ops = 0
    for op in tqdm.tqdm(ops, desc="Filtering ops by skip and synthetic ops"):
        total_ops += 1
        if any(skip_op in op["op_name"] for skip_op in SKIP_OPERATORS):
            op["included_in_benchmark"] = False
            op["why_excluded"].append("We cannot run this op on backendbench yet")
            op["runnable"] = False
            skip_ops += 1

        if op["is_synthetic"]:
            op["included_in_benchmark"] = False
            op["why_excluded"].append(
                "Synthetic ops are not supported in the official benchmark yet"
            )
            op["runnable"] = False
            synthetic_ops += 1
    return ops


def apply_runtime_filter(ops):
    # we shall define the threshold of an op being useful as taking at least
    # 3x the time of torch.randn(1) * 2 as this means it takes reasonably longer than kernel_overhead

    # Define the operation
    def _overhead_benchmark():
        return torch.randn(1, device="cuda")

    runtime_threshold_ms = do_bench(_overhead_benchmark, warmup=25, rep=100)
    runtime_threshold_ms = 2 * runtime_threshold_ms

    for op in tqdm.tqdm(ops, desc="Filtering ops by runtime"):
        if op["runnable"]:
            args, kwargs = deserialize_args(op["args"])
            try:
                op_name = op["op_name"]
                op_func = eval(f"torch.ops.{op_name}")
                ms = do_bench(lambda: op_func(*args, **kwargs), warmup=25, rep=100)
            except Exception as e:
                ms = -1
                op["why_excluded"].append(f"Failed to run: {e}")
            finally:
                del args, kwargs
                cleanup_memory_and_gpu()
        else:
            ms = -1
        op["runtime_ms"] = ms
        if ms < runtime_threshold_ms:
            op["included_in_benchmark"] = False
            op["why_excluded"].append(
                f"Runtime is too short to be meaningful. Threshhold used is {runtime_threshold_ms}ms"
            )
    return ops
