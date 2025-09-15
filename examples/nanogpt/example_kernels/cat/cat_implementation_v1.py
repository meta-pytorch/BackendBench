# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def cat_triton_kernel(
    output_ptr,
    input_ptrs,
    input_offsets,
    input_sizes,
    num_inputs: tl.constexpr,
    total_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size

    # Initialize output data
    data = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Process each input tensor
    for i in tl.static_range(num_inputs):
        input_offset = tl.load(input_offsets + i)
        input_size = tl.load(input_sizes + i)
        input_ptr = tl.load(input_ptrs + i)

        # Check if current block overlaps with this input
        input_start = input_offset
        input_end = input_offset + input_size

        # Calculate which elements in this block belong to this input
        input_mask = mask & (offsets >= input_start) & (offsets < input_end)

        # Calculate source offsets within the input tensor
        src_offsets = offsets - input_start
        src_mask = input_mask & (src_offsets >= 0) & (src_offsets < input_size)

        # Load data from input tensor
        if tl.any(src_mask):
            input_data = tl.load(input_ptr + src_offsets, mask=src_mask, other=0.0)
            data = tl.where(src_mask, input_data, data)

    # Store to output
    tl.store(output_ptr + offsets, data, mask=mask)


def cat_kernel_impl(tensors, dim=0):
    # print("Hello from cat_kernel_impl")
    if not tensors:
        raise ValueError("cat() requires at least one tensor")

    # Handle dim parameter - extract from kwargs if needed
    if isinstance(dim, torch.Tensor):
        if dim.numel() == 1:
            dim = dim.item()
        else:
            raise ValueError("dim must be a scalar")

    # Convert to integer if needed
    dim = int(dim)

    # Get device info and move tensors to GPU if needed
    original_devices = [t.device for t in tensors]
    needs_cuda = any(t.device.type == "cuda" for t in tensors)

    if needs_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Move all tensors to the same device (GPU if any tensor is on GPU)
    if needs_cuda:
        tensors = [t.cuda() if t.device.type != "cuda" else t for t in tensors]
    else:
        # If no GPU tensors, move to GPU for computation if available
        if torch.cuda.is_available():
            tensors = [t.cuda() for t in tensors]

    # Handle empty tensors by filtering them out
    non_empty_tensors = []
    for t in tensors:
        if t.numel() > 0:
            non_empty_tensors.append(t)

    if not non_empty_tensors:
        # All tensors are empty, return empty tensor with appropriate shape
        if tensors:
            result_shape = list(tensors[0].shape)
            result_shape[dim] = 0
            result = torch.empty(result_shape, dtype=tensors[0].dtype, device=tensors[0].device)
        else:
            result = torch.empty(0)

        # Move back to original device
        if original_devices and original_devices[0].type != result.device.type:
            result = result.to(original_devices[0])
        return result

    tensors = non_empty_tensors

    # Normalize dimension
    ndim = tensors[0].ndim
    if dim < 0:
        dim = dim + ndim

    if dim < 0 or dim >= ndim:
        raise IndexError(f"Dimension {dim} out of range for {ndim}D tensor")

    # Validate tensor shapes
    ref_shape = list(tensors[0].shape)
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.ndim != ndim:
            raise RuntimeError("Tensors must have same number of dimensions")

        for d in range(ndim):
            if d != dim and tensor.shape[d] != ref_shape[d]:
                raise RuntimeError(f"Sizes of tensors must match except in dimension {dim}")

    # Calculate output shape
    output_shape = ref_shape.copy()
    output_shape[dim] = sum(t.shape[dim] for t in tensors)

    # Create output tensor
    output = torch.empty(output_shape, dtype=tensors[0].dtype, device=tensors[0].device)

    # For simple cases, use torch.cat directly
    if len(tensors) <= 2 or output.numel() < 1024:
        result = torch.cat(tensors, dim=dim)
        # Move back to original device
        if original_devices and original_devices[0].type != result.device.type:
            result = result.to(original_devices[0])
        return result

    # Flatten tensors for easier processing
    # Calculate strides and reshape
    before_dim = int(torch.prod(torch.tensor(output_shape[:dim])))
    after_dim = int(torch.prod(torch.tensor(output_shape[dim + 1 :])))
    cat_dim_size = output_shape[dim]

    # Reshape tensors to 3D: [before_dim, cat_dim, after_dim]
    reshaped_tensors = []
    for tensor in tensors:
        t_before = int(torch.prod(torch.tensor(tensor.shape[:dim])))
        t_cat = tensor.shape[dim]
        t_after = int(torch.prod(torch.tensor(tensor.shape[dim + 1 :])))
        reshaped = tensor.reshape(t_before, t_cat, t_after)
        reshaped_tensors.append(reshaped)

    output_reshaped = output.reshape(before_dim, cat_dim_size, after_dim)

    # Process each slice along the first dimension
    current_offset = 0
    for tensor in reshaped_tensors:
        tensor_cat_size = tensor.shape[1]

        # Copy data slice by slice
        for i in range(before_dim):
            for j in range(after_dim):
                output_reshaped[i, current_offset : current_offset + tensor_cat_size, j] = tensor[
                    i, :, j
                ]

        current_offset += tensor_cat_size

    # Move back to original device
    if original_devices and original_devices[0].type != output.device.type:
        output = output.to(original_devices[0])

    return output
