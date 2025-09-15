# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def _unsafe_view_triton_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data from input
    data = tl.load(input_ptr + offsets, mask=mask)

    # Store data to output (same data, just different view)
    tl.store(output_ptr + offsets, data, mask=mask)


def _unsafe_view_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from _unsafe_view_kernel_impl")
    if len(args) >= 2:
        input_tensor = args[0]
        new_shape = args[1]
    else:
        input_tensor = kwargs.get("input", args[0] if args else None)
        new_shape = kwargs.get("size", args[1] if len(args) > 1 else None)

    if input_tensor is None or new_shape is None:
        raise ValueError("_unsafe_view requires input tensor and new shape")

    # Store original device
    original_device = input_tensor.device

    # Check if we need to move to GPU
    if not input_tensor.is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run Triton kernel")
        input_tensor = input_tensor.cuda()

    # Validate that the view is compatible
    original_numel = input_tensor.numel()
    if isinstance(new_shape, (list, tuple)):
        new_numel = 1
        for dim in new_shape:
            if dim == -1:
                # Handle -1 dimension inference
                inferred_size = original_numel // (new_numel if new_numel > 0 else 1)
                new_numel *= inferred_size
            else:
                new_numel *= dim
    else:
        new_numel = new_shape
        new_shape = (new_shape,)

    if new_numel != original_numel:
        raise RuntimeError(f"shape '{new_shape}' is invalid for input of size {original_numel}")

    # Create output tensor with new shape
    output_tensor = torch.empty(new_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    # Get contiguous tensors for kernel
    input_contiguous = input_tensor.contiguous()

    n_elements = input_contiguous.numel()

    if n_elements == 0:
        # Handle empty tensor case
        result = output_tensor
    else:
        # Calculate grid size
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # Launch kernel
        _unsafe_view_triton_kernel[grid](
            input_contiguous,
            output_tensor,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        result = output_tensor

    # Move result back to original device if needed
    if result.device != original_device:
        result = result.to(original_device)

    return result
