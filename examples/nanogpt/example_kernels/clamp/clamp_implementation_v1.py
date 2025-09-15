# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def clamp_triton_kernel(
    input_ptr,
    min_ptr,
    max_ptr,
    output_ptr,
    n_elements,
    has_min: tl.constexpr,
    has_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)

    # Apply clamping
    if has_min:
        min_val = tl.load(min_ptr + offsets, mask=mask)
        x = tl.maximum(x, min_val)

    if has_max:
        max_val = tl.load(max_ptr + offsets, mask=mask)
        x = tl.minimum(x, max_val)

    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)


def clamp_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from clamp_kernel_impl")
    if len(args) >= 1:
        input_tensor = args[0]
    else:
        input_tensor = kwargs.get("input", kwargs.get("self"))

    if len(args) >= 2:
        min_val = args[1]
    else:
        min_val = kwargs.get("min", None)

    if len(args) >= 3:
        max_val = args[2]
    else:
        max_val = kwargs.get("max", None)

    # Store original device
    original_device = input_tensor.device

    # Move to GPU if needed
    if not input_tensor.is_cuda and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    elif not input_tensor.is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but GPU tensor operations are required")

    # Handle min/max values
    has_min = min_val is not None
    has_max = max_val is not None

    if has_min:
        if isinstance(min_val, (int, float)):
            min_tensor = torch.full_like(input_tensor, min_val)
        else:
            min_tensor = min_val
            if not min_tensor.is_cuda and torch.cuda.is_available():
                min_tensor = min_tensor.cuda()
            # Broadcast to match input shape
            min_tensor = torch.broadcast_to(min_tensor, input_tensor.shape).contiguous()
    else:
        min_tensor = torch.zeros_like(input_tensor)  # Dummy tensor

    if has_max:
        if isinstance(max_val, (int, float)):
            max_tensor = torch.full_like(input_tensor, max_val)
        else:
            max_tensor = max_val
            if not max_tensor.is_cuda and torch.cuda.is_available():
                max_tensor = max_tensor.cuda()
            # Broadcast to match input shape
            max_tensor = torch.broadcast_to(max_tensor, input_tensor.shape).contiguous()
    else:
        max_tensor = torch.zeros_like(input_tensor)  # Dummy tensor

    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()

    # Create output tensor
    output = torch.empty_like(input_tensor)

    # Calculate grid
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    clamp_triton_kernel[grid](
        input_tensor,
        min_tensor,
        max_tensor,
        output,
        n_elements,
        has_min,
        has_max,
        BLOCK_SIZE,
    )

    # Move result back to original device if needed
    if original_device != output.device:
        output = output.to(original_device)

    return output
