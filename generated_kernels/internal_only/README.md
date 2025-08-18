# Internal PyTorch Operators

This directory contains 62 operators that don't have comprehensive PyTorch documentation available. These are typically internal or low-level operators.

## Operators in this directory:

- `_adaptive_avg_pool2d`
- `_adaptive_avg_pool2d_backward`
- `_cudnn_rnn`
- `_log_softmax_backward_data`
- `_softmax_backward_data`
- `_sparse_coo_tensor_with_dims_and_tensors`
- `_to_copy`
- `_unsafe_view`
- `add_`
- `as_strided_`
- `avg_pool2d_backward`
- `bernoulli_`
- `clamp_min`
- `convolution_backward`
- `copy_`
- `div_`
- `elu`
- `elu_backward`
- `erf`
- `fill_`
- `gelu_backward`
- `grid_sampler_2d_backward`
- `hardsigmoid_backward`
- `hardswish_backward`
- `hardtanh`
- `hardtanh_`
- `hardtanh_backward`
- `leaky_relu_`
- `leaky_relu_backward`
- `lift_fresh_copy`
- `logical_and_`
- `masked_fill`
- `masked_fill_`
- `max_pool2d_with_indices_backward`
- `mse_loss_backward`
- `mul_`
- `native_batch_norm`
- `native_batch_norm_backward`
- `native_group_norm`
- `native_group_norm_backward`
- `native_layer_norm`
- `new_empty`
- `new_empty_strided`
- `new_full`
- `new_ones`
- `new_zeros`
- `reflection_pad2d_backward`
- `relu`
- `relu_`
- `repeat`
- `rsub`
- `select_backward`
- `sigmoid`
- `sigmoid_`
- `sigmoid_backward`
- `silu_backward`
- `slice_backward`
- `split_with_sizes`
- `tanh_backward`
- `threshold_backward`
- `unfold_backward`
- `unsqueeze_`

## Implementation Notes

These operators may require:
- Examining PyTorch source code for implementation details
- Understanding internal PyTorch conventions
- More research into expected behavior

## Getting Documentation

If you find documentation for any of these operators, you can:
1. Move the directory back to `generated_kernels/`
2. Update the README.md with proper documentation
3. Update the watermarked implementation if needed

## Reference

See `internal_operators.csv` in the root directory for a complete list.
