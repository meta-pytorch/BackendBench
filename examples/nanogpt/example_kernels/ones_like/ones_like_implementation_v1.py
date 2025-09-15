# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def ones_like_triton_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Store ones to output
    tl.store(output_ptr + offsets, 1.0, mask=mask)


def ones_like_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from ones_like_kernel_impl")
    if len(args) > 0:
        input_tensor = args[0]
    elif "input" in kwargs:
        input_tensor = kwargs.pop("input")
    else:
        raise ValueError("ones_like requires an input tensor")

    # Extract dtype and device from kwargs or input tensor
    dtype = kwargs.get("dtype", input_tensor.dtype)
    device = kwargs.get("device", input_tensor.device)

    # Store original device for result
    original_device = device

    # Check if we need GPU for Triton
    needs_gpu = True
    if not torch.cuda.is_available() and needs_gpu:
        raise RuntimeError("CUDA is not available, but Triton kernel requires GPU")

    # Move input to GPU if needed for computation
    if input_tensor.device.type == "cpu" and needs_gpu:
        input_tensor_gpu = input_tensor.cuda()
        compute_device = input_tensor_gpu.device
    else:
        input_tensor_gpu = input_tensor
        compute_device = input_tensor.device

    # Create output tensor on compute device
    output_shape = input_tensor_gpu.shape
    n_elements = input_tensor_gpu.numel()

    # Create output tensor with proper dtype on compute device
    output = torch.empty(output_shape, dtype=dtype, device=compute_device)

    if n_elements == 0:
        # Handle empty tensor case
        result = output
    else:
        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        ones_like_triton_kernel[grid](
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        result = output

    # Move result back to original device if needed
    if result.device != original_device:
        result = result.to(original_device)

    return result
