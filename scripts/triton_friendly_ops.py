#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton-friendly operator configurations for KernelAgent.
"""

# Operations that work well with Triton's float-only support
# These are unary/binary operations that don't have complex dtype requirements
TRITON_FRIENDLY_OPS = [
    # Unary operations (element-wise)
    "abs",      # Absolute value
    "cos",      # Cosine
    "sin",      # Sine
    "exp",      # Exponential
    "log2",     # Logarithm base 2
    "sqrt",     # Square root
    "rsqrt",    # Reciprocal square root
    "relu",     # ReLU activation
    "sigmoid",  # Sigmoid activation
    "tanh",     # Tanh activation
    "gelu",     # GELU activation
    "elu",      # ELU activation
    "erf",      # Error function
    "reciprocal", # 1/x
    "neg",      # Negation
    "floor",    # Floor
    "round",    # Round
    
    # Binary operations (element-wise)
    "add",      # Addition
    "sub",      # Subtraction  
    "mul",      # Multiplication
    "div",      # Division
    "pow",      # Power
    "fmod",     # Floating modulo
    "remainder", # Remainder
    "maximum",  # Element-wise maximum
    "minimum",  # Element-wise minimum
    
    # Comparison operations (return bool, but operate on floats)
    "eq",       # Equal
    "ne",       # Not equal
    "lt",       # Less than
    "le",       # Less than or equal
    "gt",       # Greater than
    "ge",       # Greater than or equal
    
    # Reduction operations
    "sum",      # Sum reduction
    "mean",     # Mean reduction
    "max",      # Max reduction
    "min",      # Min reduction
    
    # Matrix operations
    "mm",       # Matrix multiplication
    "bmm",      # Batch matrix multiplication
    "addmm",    # Add matrix multiplication
    
    # Activation functions
    "hardtanh", # Hard tanh
    "_softmax", # Softmax
    "_log_softmax", # Log softmax
    "leaky_relu", # Leaky ReLU
    
    # Other operations that work well with floats
    "clone",    # Clone tensor
    "where",    # Conditional selection
    "clamp",    # Clamp values
]

# Operations that are problematic for Triton
TRITON_PROBLEMATIC_OPS = [
    # These require integer support
    "bitwise_and",
    "bitwise_not", 
    "bitwise_xor",
    
    # These are complex operations that need special handling
    "convolution",
    "convolution_backward",
    "avg_pool2d_backward",
    "_adaptive_avg_pool2d_backward",
    "max_pool2d_with_indices_backward",
    "native_group_norm_backward",
    
    # These have complex implementations
    "grid_sampler_2d",
    "upsample_bilinear2d",
    "upsample_nearest2d",
    "col2im",
    
    # These need special tensor operations
    "cat",
    "split_with_sizes",
    "repeat",
    "flip",
    "_to_copy",
    "topk",
    "nonzero",
    
    # These need careful handling
    "isinf",
    "isnan",
    "any",
    "cumsum",
    
    # Padding operations can be complex
    "constant_pad_nd",
    "reflection_pad2d",
    
    # Pooling with indices
    "max_pool2d_with_indices",
    "avg_pool2d",
    "_adaptive_avg_pool2d",
    
    # Normalization (can be done but complex)
    "native_layer_norm",
    "native_group_norm",
]

def get_triton_friendly_ops():
    """Get list of operations that work well with Triton."""
    return TRITON_FRIENDLY_OPS

def is_triton_friendly(op_name):
    """Check if an operation is Triton-friendly."""
    return op_name in TRITON_FRIENDLY_OPS

def get_float_only_test_filter():
    """Get environment variables for float-only testing."""
    # This would need to be implemented in BackendBench
    # For now, we just document what would be needed
    return {
        "BACKENDBENCH_FLOAT_ONLY": "1",
        "BACKENDBENCH_DTYPES": "float16,bfloat16,float32"
    }

if __name__ == "__main__":
    print(f"Triton-friendly operations ({len(TRITON_FRIENDLY_OPS)} ops):")
    for op in sorted(TRITON_FRIENDLY_OPS):
        print(f"  - {op}")
    
    print(f"\nProblematic operations ({len(TRITON_PROBLEMATIC_OPS)} ops):")
    for op in sorted(TRITON_PROBLEMATIC_OPS):
        print(f"  - {op}")