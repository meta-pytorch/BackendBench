# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

TENSOR_CREATION_AND_MANIPULATION_OPS = [
    "cat.default",
    "cat.out",
    "cat.names",
    "cat.names_outclone.default",
    "clone.out",
    "copy_.default",
    "elu_backward.default",
    "elu_backward.grad_input",
    "masked_fill_.Scalarmasked_fill_.Tensor",
    "new_empty.default",
    "new_empty.out",
    "new_empty_strided.default",
    "new_empty_strided.out",
    "new_full.default",
    "new_full.out",
    "new_ones.default",
    "new_ones.out",
    "new_zeros.default",
    "new_zeros.out",
    "nonzero.default",
    "nonzero.out",
    "repeat.default",
    "repeat.out",
    "split.Tensor",
    "split_with_sizes.default",
    "unsqueeze_.default",
]

RANDOM_OPS = [
    "bernoulli.default",
    "bernoulli.out",
    "bernoulli.Tensor",
    "bernoulli.Tensor_out",
]

# Operators to skip for indexing ops that need valid indices
UNSUPPORTED_OPERATORS = [
    "embedding.default",
    "embedding.out",
    "scatter.src",
    "scatter.src_out",
    "scatter.reduce",
    "scatter.reduce_out",
    "scatter.value",
    "scatter.value_out",
    "scatter.value_reduce",
    "scatter.value_reduce_outgather.default",
    "gather.out",
    "gather.dimname",
    "gather.dimname_outindex.Tensor",
    "index.Tensor_outnll_loss.default",
    "nll_loss.outim2col_backward.default",
    "im2col_backward.default",
    "im2col_backward.grad_input",
    "col2im_backward.default",
    "col2im_backward.grad_input",
    "native_layer_norm_backward.default",
    "native_layer_norm_backward.out",
    "upsample_nearest2d_backward.default",
    "upsample_nearest2d_backward.grad_input",
    "upsample_bilinear2d_backward.default",
    "upsample_bilinear2d_backward.grad_input",
    "_cudnn_rnn_backward.default",  # RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM
    "_cudnn_rnn_backward.out",
    "_fft_c2c.default",  # cuFFT only supports dimensions whose sizes are powers of two when computing in half precision
    "_fft_c2c.out",
    "_cudnn_rnn.default",  # We are running into numerical stability issues with running the forward pass multiple times
    "_cudnn_rnn.out",
]
