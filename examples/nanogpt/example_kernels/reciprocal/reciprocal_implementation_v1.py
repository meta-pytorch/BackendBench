# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def reciprocal_triton_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    result = 1.0 / x
    tl.store(output_ptr + offsets, result, mask=mask)


def reciprocal_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from reciprocal_kernel_impl")
    if args:
        input_tensor = args[0]
    else:
        input_tensor = kwargs.pop("input", None)
        if input_tensor is None:
            raise ValueError("reciprocal requires 'input' tensor")

    # Store original device
    original_device = input_tensor.device

    # Move to GPU if needed
    if not input_tensor.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run Triton kernel")
        input_tensor = input_tensor.cuda()

    # Ensure tensor is contiguous
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()

    # Create output tensor on GPU
    output = torch.empty_like(input_tensor)

    # Calculate grid and block size
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    reciprocal_triton_kernel[grid](
        input_tensor,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Move result back to original device
    if original_device != output.device:
        output = output.to(original_device)

    return output
