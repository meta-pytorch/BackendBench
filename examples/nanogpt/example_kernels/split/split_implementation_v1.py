# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def split_triton_kernel(
    input_ptr,
    output_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    split_dim,
    split_start,
    split_size,
    dim0_size,
    dim1_size,
    dim2_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)

    # Calculate total elements in output
    total_elements = dim0_size * dim1_size * dim2_size

    # Calculate starting offset for this block
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements

    # Convert linear offset to 3D coordinates for output tensor
    offset_valid = tl.where(mask, offset, 0)

    dim2_idx = offset_valid % dim2_size
    temp = offset_valid // dim2_size
    dim1_idx = temp % dim1_size
    dim0_idx = temp // dim1_size

    # Calculate input coordinates
    input_dim0_idx = dim0_idx
    input_dim1_idx = dim1_idx
    input_dim2_idx = dim2_idx

    # Adjust the coordinate for the split dimension
    if split_dim == 0:
        input_dim0_idx = dim0_idx + split_start
    elif split_dim == 1:
        input_dim1_idx = dim1_idx + split_start
    elif split_dim == 2:
        input_dim2_idx = dim2_idx + split_start

    # Calculate input and output addresses
    input_offset = (
        input_dim0_idx * input_stride_0
        + input_dim1_idx * input_stride_1
        + input_dim2_idx * input_stride_2
    )

    output_offset = (
        dim0_idx * output_stride_0 + dim1_idx * output_stride_1 + dim2_idx * output_stride_2
    )

    # Load from input and store to output
    data = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + output_offset, data, mask=mask)


def split_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from split_kernel_impl")
    if len(args) >= 1:
        input_tensor = args[0]
    else:
        input_tensor = kwargs.get("input", kwargs.get("tensor"))

    if len(args) >= 2:
        split_size_or_sections = args[1]
    else:
        split_size_or_sections = kwargs.get("split_size_or_sections", kwargs.get("split_size"))

    if len(args) >= 3:
        dim = args[2]
    else:
        dim = kwargs.get("dim", 0)

    # Handle device management
    original_device = input_tensor.device
    if input_tensor.device.type == "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot move tensor to GPU")
        input_tensor = input_tensor.cuda()

    # Ensure tensor is contiguous
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()

    # Normalize dimension
    if dim < 0:
        dim = input_tensor.ndim + dim

    # Handle different input shapes by padding to 3D
    original_shape = input_tensor.shape
    if input_tensor.ndim == 1:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        if dim == 0:
            dim = 2
    elif input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(0)
        if dim >= 1:
            dim = dim + 1
    elif input_tensor.ndim > 3:
        # For higher dimensions, reshape to 3D
        shape = input_tensor.shape
        if dim == 0:
            # Keep first dimension, flatten others
            input_tensor = input_tensor.reshape(shape[0], -1, 1)
            dim = 0
        elif dim == len(shape) - 1:
            # Keep last dimension, flatten others
            input_tensor = input_tensor.reshape(-1, 1, shape[-1])
            dim = 2
        else:
            # Keep the split dimension, flatten before and after
            before_size = 1
            for i in range(dim):
                before_size *= shape[i]
            after_size = 1
            for i in range(dim + 1, len(shape)):
                after_size *= shape[i]
            input_tensor = input_tensor.reshape(before_size, shape[dim], after_size)
            dim = 1

    # Get tensor properties
    input_shape = input_tensor.shape
    dim_size = input_shape[dim]

    # Handle split_size_or_sections
    if isinstance(split_size_or_sections, int):
        # Split into chunks of given size
        split_size = split_size_or_sections
        num_splits = (dim_size + split_size - 1) // split_size  # Ceiling division
        split_sizes = [split_size] * (num_splits - 1)
        if dim_size % split_size != 0:
            split_sizes.append(dim_size % split_size)
        else:
            split_sizes.append(split_size)
    else:
        # split_size_or_sections is a list of sizes
        split_sizes = split_size_or_sections
        num_splits = len(split_sizes)

    # Create output tensors
    outputs = []
    current_start = 0

    for split_size in split_sizes:
        # Calculate output shape
        output_shape = list(input_shape)
        output_shape[dim] = split_size

        # Create output tensor
        output_tensor = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        # Calculate grid size
        total_elements = output_tensor.numel()
        BLOCK_SIZE = 1024
        grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Get strides
        input_strides = input_tensor.stride()
        output_strides = output_tensor.stride()

        # Pad strides to 3D if needed
        while len(input_strides) < 3:
            input_strides = input_strides + (1,)
        while len(output_strides) < 3:
            output_strides = output_strides + (1,)

        # Launch kernel
        split_triton_kernel[(grid_size,)](
            input_tensor,
            output_tensor,
            input_strides[0],
            input_strides[1],
            input_strides[2],
            output_strides[0],
            output_strides[1],
            output_strides[2],
            dim,
            current_start,
            split_size,
            output_shape[0] if len(output_shape) > 0 else 1,
            output_shape[1] if len(output_shape) > 1 else 1,
            output_shape[2] if len(output_shape) > 2 else 1,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Reshape output back to original dimensionality
        if len(original_shape) == 1:
            if dim == 0:  # Was split along the only dimension
                target_shape = (split_size,)
            else:
                target_shape = original_shape
            output_tensor = output_tensor.reshape(target_shape)
        elif len(original_shape) == 2:
            if dim == 0:
                target_shape = (split_size, original_shape[1])
            elif dim == 1:
                target_shape = (original_shape[0], split_size)
            else:
                target_shape = original_shape
            output_tensor = output_tensor.reshape(target_shape)
        elif len(original_shape) > 3:
            # Reshape back to original dimensionality
            target_shape = list(original_shape)
            target_shape[dim if dim < len(original_shape) else len(original_shape) - 1] = split_size
            output_tensor = output_tensor.reshape(target_shape)

        # Move back to original device if needed
        if original_device.type == "cpu":
            output_tensor = output_tensor.cpu()

        outputs.append(output_tensor)
        current_start += split_size

    return outputs
