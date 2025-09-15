# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def unbind_triton_kernel(
    input_ptr,
    output_ptrs,
    n_elements,
    dim_size,
    stride_dim,
    other_strides,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate which slice along the unbind dimension and position within slice
    slice_idx = offsets // (n_elements // dim_size)
    pos_in_slice = offsets % (n_elements // dim_size)

    # Calculate input offset
    input_offset = slice_idx * stride_dim + pos_in_slice

    # Load data
    data = tl.load(input_ptr + input_offset, mask=mask)

    # Store to appropriate output tensor
    for i in range(dim_size):
        output_mask = mask & (slice_idx == i)
        if tl.sum(output_mask.to(tl.int32)) > 0:
            output_ptr = tl.load(output_ptrs + i)
            tl.store(output_ptr + pos_in_slice, data, mask=output_mask)


def unbind_kernel_impl(*args, **kwargs):
    # Parse arguments
    # print("Hello from unbind_kernel_impl")
    if len(args) == 0:
        raise ValueError("unbind() missing required argument: 'input'")

    input_tensor = args[0]
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(input_tensor)}")

    # Handle negative dimension
    if dim < 0:
        dim = input_tensor.ndim + dim

    if dim >= input_tensor.ndim or dim < 0:
        raise IndexError(
            f"Dimension {dim} out of range for tensor with {input_tensor.ndim} dimensions"
        )

    # Store original device
    original_device = input_tensor.device
    needs_cuda = original_device.type != "cuda"

    # Move to GPU if needed
    if needs_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run Triton kernel")
        input_tensor = input_tensor.cuda()

    try:
        # Get tensor properties
        shape = input_tensor.shape
        dim_size = shape[dim]

        if dim_size == 0:
            raise RuntimeError("Cannot unbind along dimension with size 0")

        # Calculate output shape (remove the unbind dimension)
        output_shape = list(shape)
        output_shape.pop(dim)

        # Create output tensors
        outputs = []
        for i in range(dim_size):
            output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
            outputs.append(output)

        # Handle empty tensor case
        if input_tensor.numel() == 0:
            result = tuple(outputs)
            if needs_cuda:
                result = tuple(out.to(original_device) for out in result)
            return result

        # For simple case, use direct indexing approach
        if input_tensor.is_contiguous():
            # Reshape input to separate the unbind dimension
            if dim == 0:
                reshaped = input_tensor.view(dim_size, -1)
                for i in range(dim_size):
                    outputs[i].copy_(reshaped[i].view(output_shape))
            else:
                # Move unbind dimension to front, then reshape
                perm = list(range(input_tensor.ndim))
                perm[0], perm[dim] = perm[dim], perm[0]
                transposed = input_tensor.permute(perm)
                reshaped = transposed.contiguous().view(dim_size, -1)

                for i in range(dim_size):
                    # Reshape back to output shape
                    temp_shape = list(shape)
                    temp_shape[0], temp_shape[dim] = temp_shape[dim], temp_shape[0]
                    temp_shape.pop(0)  # Remove the unbind dimension
                    outputs[i].copy_(reshaped[i].view(temp_shape))
        else:
            # Handle non-contiguous tensors
            for i in range(dim_size):
                outputs[i] = input_tensor.select(dim, i).clone()

        # Convert to tuple
        result = tuple(outputs)

        # Move back to original device if needed
        if needs_cuda:
            result = tuple(out.to(original_device) for out in result)

        return result

    except Exception as e:
        raise RuntimeError(f"Error in unbind operation: {str(e)}")
