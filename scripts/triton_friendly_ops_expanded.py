#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Expanded Triton-friendly operator configurations for KernelAgent.
Based on analysis of all 143 TorchBench operations.
"""

# Operations that work well with Triton's float-only support
# Expanded from all 143 TorchBench operations
TRITON_FRIENDLY_OPS_EXPANDED = [
    # === Unary operations (element-wise) ===
    "abs",          # Absolute value
    "cos",          # Cosine
    "sin",          # Sine  
    "exp",          # Exponential
    "log2",         # Logarithm base 2
    "sqrt",         # Square root
    "rsqrt",        # Reciprocal square root
    "reciprocal",   # 1/x
    "neg",          # Negation
    "floor",        # Floor
    "round",        # Round
    "erf",          # Error function
    "sgn",          # Sign function
    
    # === Activation functions ===
    "relu",         # ReLU activation
    "relu_",        # In-place ReLU
    "sigmoid",      # Sigmoid activation
    "sigmoid_",     # In-place sigmoid
    "tanh",         # Tanh activation
    "gelu",         # GELU activation
    "elu",          # ELU activation
    "silu",         # SiLU/Swish activation
    "silu_",        # In-place SiLU
    "hardtanh",     # Hard tanh
    "hardtanh_",    # In-place hard tanh
    "hardsigmoid",  # Hard sigmoid
    "hardswish",    # Hard swish
    "hardswish_",   # In-place hard swish
    "leaky_relu",   # Leaky ReLU
    "leaky_relu_",  # In-place leaky ReLU
    "_softmax",     # Softmax
    "_log_softmax", # Log softmax
    
    # === Binary operations (element-wise) ===
    "add",          # Addition
    "add_",         # In-place addition
    "sub",          # Subtraction
    "rsub",         # Reverse subtraction (b - a)
    "mul",          # Multiplication
    "mul_",         # In-place multiplication
    "div",          # Division
    "div_",         # In-place division
    "pow",          # Power
    "fmod",         # Floating modulo
    "remainder",    # Remainder
    "maximum",      # Element-wise maximum
    "minimum",      # Element-wise minimum
    "floor_divide", # Floor division
    
    # === Ternary operations ===
    "addcmul",      # a + alpha * b * c
    "where",        # Conditional selection
    "clamp",        # Clamp values
    "clamp_min",    # Clamp minimum only
    
    # === Comparison operations ===
    "eq",           # Equal
    "ne",           # Not equal
    "lt",           # Less than
    "le",           # Less than or equal
    "gt",           # Greater than
    "ge",           # Greater than or equal
    
    # === Reduction operations ===
    "sum",          # Sum reduction
    "mean",         # Mean reduction
    "max",          # Max reduction
    "min",          # Min reduction
    "norm",         # Norm computation
    "std",          # Standard deviation
    "var_mean",     # Variance and mean
    
    # === Matrix operations ===
    "mm",           # Matrix multiplication
    "bmm",          # Batch matrix multiplication
    "addmm",        # Add matrix multiplication
    
    # === Backward operations (gradients) ===
    "sigmoid_backward",     # Sigmoid gradient
    "tanh_backward",        # Tanh gradient
    "elu_backward",         # ELU gradient
    "gelu_backward",        # GELU gradient
    "hardtanh_backward",    # Hard tanh gradient
    "hardsigmoid_backward", # Hard sigmoid gradient
    "hardswish_backward",   # Hard swish gradient
    "leaky_relu_backward",  # Leaky ReLU gradient
    "silu_backward",        # SiLU gradient
    "threshold_backward",   # Threshold gradient
    "_softmax_backward_data",     # Softmax gradient
    "_log_softmax_backward_data", # Log softmax gradient
    
    # === Loss functions ===
    "mse_loss",             # Mean squared error
    "mse_loss_backward",    # MSE gradient
    
    # === Other simple operations ===
    "clone",        # Clone tensor
    "fill_",        # Fill with value
    "masked_fill",  # Masked fill
    "masked_fill_", # In-place masked fill
    "tril",         # Lower triangular
    "triu",         # Upper triangular
]

# Operations that are problematic for Triton
TRITON_PROBLEMATIC_OPS_EXPANDED = [
    # === Integer-specific operations ===
    "bitwise_and",
    "bitwise_not",
    "bitwise_xor",
    "logical_and_",
    
    # === Complex convolution/pooling ===
    "convolution",
    "convolution_backward",
    "avg_pool2d",
    "avg_pool2d_backward",
    "_adaptive_avg_pool2d",
    "_adaptive_avg_pool2d_backward",
    "max_pool2d_with_indices",
    "max_pool2d_with_indices_backward",
    "grid_sampler_2d",
    "grid_sampler_2d_backward",
    "upsample_bilinear2d",
    "upsample_bicubic2d",
    "upsample_nearest2d",
    
    # === Tensor manipulation (complex memory patterns) ===
    "cat",
    "stack",
    "split",
    "split_with_sizes",
    "unbind",
    "repeat",
    "roll",
    "flip",
    "_to_copy",
    "as_strided_",
    "_unsafe_view",
    "lift_fresh_copy",
    "copy_",
    
    # === Special tensor operations ===
    "nonzero",
    "topk",
    "cumsum",
    "any",
    "isinf",
    "isnan",
    
    # === Padding operations ===
    "constant_pad_nd",
    "reflection_pad2d",
    "reflection_pad2d_backward",
    "col2im",
    "im2col",
    
    # === Normalization (complex) ===
    "native_layer_norm",
    "native_group_norm",
    "native_group_norm_backward",
    "native_batch_norm",
    "native_batch_norm_backward",
    
    # === Special operations ===
    "_cudnn_rnn",
    "_sparse_coo_tensor_with_dims_and_tensors",
    "bernoulli_",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones", 
    "new_zeros",
    "unsqueeze_",
    
    # === Complex backward operations ===
    "select_backward",
    "slice_backward",
    "unfold_backward",
]

def get_triton_friendly_ops_expanded():
    """Get expanded list of operations that work well with Triton."""
    return TRITON_FRIENDLY_OPS_EXPANDED

def get_triton_problematic_ops_expanded():
    """Get expanded list of operations that are problematic for Triton."""
    return TRITON_PROBLEMATIC_OPS_EXPANDED

def is_triton_friendly_expanded(op_name):
    """Check if an operation is Triton-friendly."""
    return op_name in TRITON_FRIENDLY_OPS_EXPANDED

if __name__ == "__main__":
    print(f"Triton-friendly operations ({len(TRITON_FRIENDLY_OPS_EXPANDED)} ops):")
    for i, op in enumerate(sorted(TRITON_FRIENDLY_OPS_EXPANDED), 1):
        print(f"  {i:3d}. {op}")
    
    print(f"\nProblematic operations ({len(TRITON_PROBLEMATIC_OPS_EXPANDED)} ops):")
    for i, op in enumerate(sorted(TRITON_PROBLEMATIC_OPS_EXPANDED), 1):
        print(f"  {i:3d}. {op}")
    
    # Verify coverage
    total_categorized = len(TRITON_FRIENDLY_OPS_EXPANDED) + len(TRITON_PROBLEMATIC_OPS_EXPANDED)
    print(f"\nTotal categorized: {total_categorized}/143 TorchBench operations")