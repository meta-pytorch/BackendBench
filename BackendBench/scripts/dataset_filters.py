# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

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

UNTESTABLE_OPERATORS = [
    "empty_like",  # We can check using metadata
    "new_empty",  # We can check using metadata
    "new_empty_strided",  # We can check using metadata
    "bernoulli",  # We can write a custom test to verify this one (albeit not the randomness)
]


def apply_skip_ops_filter(ops):
    for op in ops:
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
    return ops
