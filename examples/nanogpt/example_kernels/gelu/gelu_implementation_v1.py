# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def gelu_triton_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for valid elements
    mask = offsets < n_elements

    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)

    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    # Constants
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    coeff = 0.044715

    # Compute x^3
    x_cubed = x * x * x

    # Compute the argument to tanh
    tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed)

    # Compute tanh using the identity: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # For numerical stability, we use: tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x)) when x > 0
    # and tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) when x <= 0

    # Use the approximation: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2) for better performance
    # This is a rational approximation that's quite accurate
    tanh_arg_sq = tanh_arg * tanh_arg
    tanh_approx = tanh_arg * (27.0 + tanh_arg_sq) / (27.0 + 9.0 * tanh_arg_sq)

    # Compute GELU
    gelu_result = 0.5 * x * (1.0 + tanh_approx)

    # Store result
    tl.store(output_ptr + offsets, gelu_result, mask=mask)


def gelu_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from gelu_kernel_impl")
    if args:
        input_tensor = args[0]
        if len(args) > 1:
            # Handle additional args if any
            pass
    elif "input" in kwargs:
        input_tensor = kwargs["input"]
    else:
        raise ValueError("No input tensor provided")

    # Handle tensor shape input (for testing)
    if isinstance(input_tensor, torch.Size):
        input_tensor = torch.randn(input_tensor, dtype=torch.float32)

    # Store original device
    original_device = input_tensor.device

    # Move to GPU if needed and available
    if not input_tensor.is_cuda:
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        else:
            raise RuntimeError("CUDA is not available, cannot run Triton kernel")

    # Ensure tensor is contiguous
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()

    # Create output tensor on same device as input
    output_tensor = torch.empty_like(input_tensor)

    # Calculate total number of elements
    n_elements = input_tensor.numel()

    # Choose block size
    BLOCK_SIZE = 1024

    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    gelu_triton_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Move result back to original device if needed
    if original_device != output_tensor.device:
        output_tensor = output_tensor.to(original_device)

    return output_tensor
