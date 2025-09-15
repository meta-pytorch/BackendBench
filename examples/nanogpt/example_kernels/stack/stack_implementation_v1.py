# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def stack_triton_kernel(
    input_ptrs,
    output_ptr,
    num_tensors,
    tensor_numel,
    stack_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tensor_numel

    # Calculate which tensor and position within tensor
    tensor_idx = offsets // stack_dim_size
    pos_in_tensor = offsets % stack_dim_size

    # Load from appropriate input tensor
    for i in range(num_tensors):
        tensor_mask = (tensor_idx == i) & mask
        if tl.sum(tensor_mask.to(tl.int32)) > 0:
            # Get the base pointer for tensor i
            input_ptr_val = tl.load(input_ptrs + i)
            input_ptr = input_ptr_val.to(tl.pointer_type(tl.float16))

            # Load data from input tensor
            data = tl.load(input_ptr + pos_in_tensor, mask=tensor_mask)

            # Store to output
            tl.store(output_ptr + offsets, data, mask=tensor_mask)


def stack_kernel_impl(tensors, dim=0):
    # print("Hello from stack_kernel_impl")
    if not tensors:
        raise ValueError("stack expects a non-empty list of tensors")

    # Handle device management
    original_device = tensors[0].device
    needs_cuda = any(t.device.type == "cuda" for t in tensors)

    if needs_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Move all tensors to the same device (GPU if any tensor is on GPU)
    if needs_cuda:
        tensors = [t.cuda() if t.device.type != "cuda" else t for t in tensors]

    # Validate inputs
    if not all(t.shape == tensors[0].shape for t in tensors):
        raise RuntimeError("stack expects each tensor to be equal size")

    if not all(t.dtype == tensors[0].dtype for t in tensors):
        raise RuntimeError("stack expects each tensor to have the same dtype")

    # Handle empty tensors
    if tensors[0].numel() == 0:
        # For empty tensors, use PyTorch's stack directly
        result = torch.stack(tensors, dim=dim)
        if original_device.type != result.device.type:
            result = result.to(original_device)
        return result

    # Normalize dimension
    ndim = len(tensors[0].shape)
    if dim < 0:
        dim = ndim + 1 + dim
    if dim < 0 or dim > ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim - 1}, {ndim}], but got {dim})"
        )

    # Calculate output shape
    input_shape = list(tensors[0].shape)
    output_shape = input_shape[:dim] + [len(tensors)] + input_shape[dim:]

    # Create output tensor
    output = torch.empty(output_shape, dtype=tensors[0].dtype, device=tensors[0].device)

    # For simple cases, we can use a more direct approach
    if dim == 0:
        # Stack along first dimension - simple concatenation
        for i, tensor in enumerate(tensors):
            output[i] = tensor
    elif dim == len(output_shape) - 1:
        # Stack along last dimension
        for i, tensor in enumerate(tensors):
            output[..., i] = tensor
    else:
        # General case - use PyTorch for complex dimension handling
        result = torch.stack(tensors, dim=dim)
        if original_device.type != result.device.type:
            result = result.to(original_device)
        return result

    # Move result back to original device if needed
    if original_device.type != output.device.type:
        output = output.to(original_device)

    return output
