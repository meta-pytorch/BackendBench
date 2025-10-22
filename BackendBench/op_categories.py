# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

TENSOR_CREATION_AND_MANIPULATION_OPS = [
    "cat.default",
    "clone.default",
    # "copy_",
    # "elu_backward",
    # "masked_fill_",
    "new_empty.default",
    "new_empty_strided.default",
    "new_full.default",
    "new_ones.default",
    "new_zeros.default",
    "nonzero.default",
    "repeat.default",
    "split.Tensor",
    "split_with_sizes.default",
    # "unsqueeze_",
]

RANDOM_OPS = [
    "bernoulli.default",
]

# Operators to skip for indexing ops that need valid indices
UNSUPPORTED_OPERATORS = [
    "embedding.default",
    "scatter.src",
    "scatter.reduce",
    "scatter.value",
    "scatter.value_reduce",
    "gather.default",
    "index.Tensor",
    # "nll_loss",
    # "im2col_backward",
    # "col2im_backward",
    # "native_layer_norm_backward",
    "upsample_nearest2d_backward.vec",
    "upsample_bilinear2d_backward.vec",
    "_cudnn_rnn_backward.default",  # RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM
    "_fft_c2c.default",  # cuFFT only supports dimensions whose sizes are powers of two when computing in half precision
    "_cudnn_rnn.default",  # We are running into numerical stability issues with running the forward pass multiple times
]
