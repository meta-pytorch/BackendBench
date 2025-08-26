# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

TENSOR_CREATION_AND_MANIPULATION_OPS = [
    "arange",
    "as_strided",  # broken in Shahin's run
    "as_strided_copy",
    "as_strided_scatter",  # broken in Shahin's run
    "cat",  # broken in Shahin's run
    "clone",
    "copy",
    "detach",
    "diagonal_copy",  # broken in Shahin's run
    "empty",
    "empty_like",
    "empty_strided",
    "expand",
    "expand_copy",
    "full",  # broken in Shahin's run
    "full_like",  # broken in Shahin's run
    "linspace",
    "new_empty",  # broken in Shahin's run
    "new_empty_strided",  # broken in Shahin's run
    "new_full",  # broken in Shahin's run
    "new_ones",  # broken in Shahin's run
    "new_zeros",  # broken in Shahin's run
    "nextafter",  # broken in Shahin's run
    "nonzero",  # broken in Shahin's run
    "ones",  # broken in Shahin's run
    "ones_like",  # broken in Shahin's run
    "permute",  # broken in Shahin's run
    "permute_copy",  # broken in Shahin's run
    "pixel_shuffle",  # broken in Shahin's run
    "pixel_unshuffle",  # broken in Shahin's run
    "randint_like",  # broken in Shahin's run
    "randn_like",  # broken in Shahin's run
    "repeat",  # broken in Shahin's run
    "resize",
    "resize_as",  # broken in Shahin's run
    "split",
    "split_with_sizes",
    "split_with_sizes_copy",
    "t",
    "t_copy",
    "to_copy",  # broken in Shahin's run
    "to_sparse",  # broken in Shahin's run
    "view",
    "view_as_complex",
    "zeros",  # broken in Shahin's run
    "zeros_like",  # broken in Shahin's run
]

RANDOM_OPS = [
    "bernoulli",  # broken in Shahin's run
    "cauchy",
    "exponential",
    "geometric",
    "log_normal",
    "multinomial",
    "normal",
    "randint",  # broken in Shahin's run
    "randint_like",  # broken in Shahin's run
    "randn_like",  # broken in Shahin's run
    "renorm",  # broken in Shahin's run
    "uniform",
]

UNTESTABLE_OPERATORS = [
    "empty_like",  # We can check using metadata
    "new_empty",  # We can check using metadata
    "new_empty_strided",  # We can check using metadata
    "bernoulli",  # We can write a custom test to verify this one (albeit not the randomness)
]

# Operators to skip for indexing ops that need valid indices
UNSUPPORTED_OPERATORS = [
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
