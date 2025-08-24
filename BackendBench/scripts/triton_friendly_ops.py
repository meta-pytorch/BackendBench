#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton operator classification for KernelAgent.

Based on compiler analysis, operations are classified into three categories:
1. Triton-friendly: Static tiled loops with affine index maps, good performance expected
2. Triton-capable: Doable but requires careful engineering or has performance caveats
3. Triton-challenging: Genuinely problematic due to hardware/compiler limitations
"""

# ✅ TRITON-FRIENDLY: Easy wins with good expected performance
# These ops have static tiled loop nests, affine index maps, coalesced access patterns
TRITON_FRIENDLY_OPS = [
    # === Unary operations (element-wise) ===
    "abs",  # Absolute value
    "cos",  # Cosine
    "sin",  # Sine
    "exp",  # Exponential
    "log2",  # Logarithm base 2
    "sqrt",  # Square root
    "rsqrt",  # Reciprocal square root
    "reciprocal",  # 1/x
    "neg",  # Negation
    "floor",  # Floor
    "round",  # Round
    "erf",  # Error function
    "sgn",  # Sign function
    
    # === Activation functions ===
    "relu",  # ReLU activation
    "relu_",  # In-place ReLU
    "sigmoid",  # Sigmoid activation
    "sigmoid_",  # In-place sigmoid
    "tanh",  # Tanh activation
    "gelu",  # GELU activation
    "elu",  # ELU activation
    "silu",  # SiLU/Swish activation
    "silu_",  # In-place SiLU
    "hardtanh",  # Hard tanh
    "hardtanh_",  # In-place hard tanh
    "hardsigmoid",  # Hard sigmoid
    "hardswish",  # Hard swish
    "hardswish_",  # In-place hard swish
    "leaky_relu",  # Leaky ReLU
    "leaky_relu_",  # In-place leaky ReLU
    "_softmax",  # Softmax (single-axis reduction)
    "_log_softmax",  # Log softmax (single-axis reduction)
    
    # === Binary operations (element-wise) ===
    "add",  # Addition
    "add_",  # In-place addition
    "sub",  # Subtraction
    "rsub",  # Reverse subtraction (b - a)
    "mul",  # Multiplication
    "mul_",  # In-place multiplication
    "div",  # Division (float)
    "div_",  # In-place division
    "pow",  # Power (prefer float base/exp)
    "maximum",  # Element-wise maximum
    "minimum",  # Element-wise minimum
    
    # === Ternary operations ===
    "addcmul",  # a + alpha * b * c
    "where",  # Conditional selection (with masks)
    "clamp",  # Clamp values
    "clamp_min",  # Clamp minimum only
    
    # === Comparison operations ===
    "eq",  # Equal
    "ne",  # Not equal
    "lt",  # Less than
    "le",  # Less than or equal
    "gt",  # Greater than
    "ge",  # Greater than or equal
    "isinf",  # Check for infinity (element-wise)
    "isnan",  # Check for NaN (element-wise)
    
    # === Simple reductions (single-axis) ===
    "sum",  # Sum reduction
    "mean",  # Mean reduction
    "max",  # Max reduction
    "min",  # Min reduction
    "std",  # Standard deviation (single-axis)
    "var_mean",  # Variance and mean (single-axis)
    "any",  # Any true (reduction)
    
    # === Regular matrix operations ===
    "mm",  # Matrix multiplication
    "bmm",  # Batch matrix multiplication
    "addmm",  # Add matrix multiplication (C + A @ B)
    
    # === Backward operations (element-wise gradients) ===
    "sigmoid_backward",  # Sigmoid gradient
    "tanh_backward",  # Tanh gradient
    "elu_backward",  # ELU gradient
    "gelu_backward",  # GELU gradient
    "hardtanh_backward",  # Hard tanh gradient
    "hardsigmoid_backward",  # Hard sigmoid gradient
    "hardswish_backward",  # Hard swish gradient
    "leaky_relu_backward",  # Leaky ReLU gradient
    "silu_backward",  # SiLU gradient
    "threshold_backward",  # Threshold gradient
    
    # === Simple loss functions ===
    "mse_loss",  # Mean squared error (element-wise + reduction)
    "mse_loss_backward",  # MSE gradient
    
    # === Bitwise operations (int32 preferred) ===
    "bitwise_and",  # Bitwise AND (int32)
    "bitwise_xor",  # Bitwise XOR (int32)
    "bitwise_not",  # Bitwise NOT (int32)
    "logical_and_",  # Logical AND (int32)
    
    # === Simple memory operations ===
    "clone",  # Clone tensor (simple copy)
    "copy_",  # In-place copy
    "fill_",  # Fill with value
    "masked_fill",  # Masked fill (with affine masks)
    "masked_fill_",  # In-place masked fill
    "tril",  # Lower triangular (affine indexing)
    "triu",  # Upper triangular (affine indexing)
    "unsqueeze_",  # In-place unsqueeze (simple shape change)
]

# ⚠️ TRITON-CAPABLE: Doable but requires careful engineering
# These ops can be implemented efficiently but need attention to tiling, shared memory, atomics
TRITON_CAPABLE_OPS = [
    # === Multi-axis/global reductions ===
    "norm",  # Norm computation (may need multi-pass)
    "_softmax_backward_data",  # Softmax gradient (reduction + broadcast)
    "_log_softmax_backward_data",  # Log softmax gradient
    
    # === Convolution/pooling (engineering-heavy but doable) ===
    "convolution",  # Can be done with careful SMEM tiling
    "convolution_backward",  # Gradient convolution
    "avg_pool2d",  # Average pooling
    "avg_pool2d_backward",  # Average pooling backward
    "_adaptive_avg_pool2d",  # Adaptive average pooling
    "_adaptive_avg_pool2d_backward",  # Adaptive average pooling backward
    "max_pool2d_with_indices",  # Max pooling with indices
    "max_pool2d_with_indices_backward",  # Max pooling backward
    
    # === Backward operations (need gradient computation) ===
    "grid_sampler_2d_backward",  # Grid sampler backward
    "reflection_pad2d_backward",  # Reflection padding backward
    "select_backward",  # Select backward
    "slice_backward",  # Slice backward
    "unfold_backward",  # Unfold backward
    
    # === Normalization (requires atomics for training) ===
    "native_layer_norm",  # Layer norm (reduction + broadcast)
    "native_group_norm",  # Group norm
    "native_group_norm_backward",  # Group norm backward
    "native_batch_norm",  # Batch norm (training needs atomics)
    "native_batch_norm_backward",  # BN gradients
    
    # === Integer operations (prefer int32) ===
    "floor_divide",  # Integer division (slower than float ops)
    "fmod",  # Floating modulo
    "remainder",  # Integer remainder
    
    # === Tensor manipulation (depends on layout) ===
    "cat",  # Concatenation (OK if contiguous)
    "stack",  # Stack (OK if regular strides)
    "split",  # Split (OK if even splits)
    "repeat",  # Repeat (OK if affine pattern)
    
    # === Indexing operations (performance varies) ===
    # Note: Removed index, index_put, scatter, gather as they're not in TorchBench
    
    # === Special operations ===
    "grid_sampler_2d",  # Bilinear sampling (careful indexing)
    "upsample_bilinear2d",  # Bilinear upsampling
    "upsample_bicubic2d",  # Bicubic upsampling
    "upsample_nearest2d",  # Nearest neighbor upsampling
    "constant_pad_nd",  # Constant padding (affine if regular)
    "bernoulli_",  # RNG via Philox counters
    # Note: Removed dropout as it's not in TorchBench
]

# ❌ TRITON-CHALLENGING: Genuinely problematic operations
# These hit fundamental limitations or require features Triton doesn't handle well
TRITON_CHALLENGING_OPS = [
    # === Int64-heavy arithmetic ===
    "cumsum",  # Cumulative sum (often int64 indices)
    # Note: Removed cumprod as it's not in TorchBench
    
    # === Highly dynamic/irregular ops ===
    "nonzero",  # Dynamic output size
    # Note: Removed unique as it's not in TorchBench
    "topk",  # Data-dependent sorting
    
    # === Complex memory patterns ===
    "as_strided_",  # Arbitrary striding
    "_unsafe_view",  # Unsafe view operations
    # Note: Removed unfold as it's not in TorchBench
    "roll",  # Circular shift (non-affine)
    "flip",  # Reverse dimensions
    
    # === Ragged/variable operations ===
    "split_with_sizes",  # Variable size splits
    "unbind",  # Unbind into list
    # Note: Removed nested_tensor as it's not in TorchBench
    
    # === Special tensor types ===
    "_sparse_coo_tensor_with_dims_and_tensors",  # Sparse ops
    "_to_copy",  # Complex dtype/device copies
    
    # === Dynamic tensor creation ===
    "lift_fresh_copy",  # Creates new tensor copies
    "new_empty",  # Dynamic tensor creation
    "new_empty_strided",  # Dynamic strided tensor creation
    "new_full",  # Dynamic tensor creation with fill
    "new_ones",  # Dynamic tensor creation (ones)
    "new_zeros",  # Dynamic tensor creation (zeros)
    
    # === Multi-device/distributed ===
    # Note: Removed _c10d_functional and all_reduce as they're not in TorchBench
    
    # === Very complex patterns ===
    "_cudnn_rnn",  # Complex RNN implementations
    "reflection_pad2d",  # Reflection padding (complex indexing)
    "col2im",  # Complex layout transformation
    "im2col",  # Complex layout transformation
    
    # === Dynamic control flow ===
    # Note: Removed cond and while_loop as they're not in TorchBench
]


def get_triton_friendly_ops():
    """Get list of operations that work well with Triton."""
    return TRITON_FRIENDLY_OPS


def get_triton_capable_ops():
    """Get list of operations that can be done in Triton with effort."""
    return TRITON_CAPABLE_OPS


def get_triton_challenging_ops():
    """Get list of operations that are genuinely problematic for Triton."""
    return TRITON_CHALLENGING_OPS


def classify_operation(op_name):
    """Classify an operation as friendly, capable, or challenging."""
    if op_name in TRITON_FRIENDLY_OPS:
        return "friendly"
    elif op_name in TRITON_CAPABLE_OPS:
        return "capable"
    elif op_name in TRITON_CHALLENGING_OPS:
        return "challenging"
    else:
        return "unknown"


# For backward compatibility
TRITON_FRIENDLY_OPS_EXPANDED = TRITON_FRIENDLY_OPS
TRITON_PROBLEMATIC_OPS_EXPANDED = TRITON_CAPABLE_OPS + TRITON_CHALLENGING_OPS


if __name__ == "__main__":
    print(f"✅ Triton-friendly operations ({len(TRITON_FRIENDLY_OPS)} ops):")
    print("   Easy wins with good expected performance")
    for i, op in enumerate(sorted(TRITON_FRIENDLY_OPS), 1):
        print(f"   {i:3d}. {op}")
    
    print(f"\n⚠️  Triton-capable operations ({len(TRITON_CAPABLE_OPS)} ops):")
    print("   Doable but requires careful engineering")
    for i, op in enumerate(sorted(TRITON_CAPABLE_OPS), 1):
        print(f"   {i:3d}. {op}")
    
    print(f"\n❌ Triton-challenging operations ({len(TRITON_CHALLENGING_OPS)} ops):")
    print("   Genuinely problematic due to limitations")
    for i, op in enumerate(sorted(TRITON_CHALLENGING_OPS), 1):
        print(f"   {i:3d}. {op}")
    
    # Summary
    total_ops = len(TRITON_FRIENDLY_OPS) + len(TRITON_CAPABLE_OPS) + len(TRITON_CHALLENGING_OPS)
    print(f"\nTotal categorized: {total_ops} operations")
    print(f"Friendly: {len(TRITON_FRIENDLY_OPS)} ({len(TRITON_FRIENDLY_OPS)/total_ops*100:.1f}%)")
    print(f"Capable: {len(TRITON_CAPABLE_OPS)} ({len(TRITON_CAPABLE_OPS)/total_ops*100:.1f}%)")
    print(f"Challenging: {len(TRITON_CHALLENGING_OPS)} ({len(TRITON_CHALLENGING_OPS)/total_ops*100:.1f}%)")