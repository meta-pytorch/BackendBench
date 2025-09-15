# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def view_triton_kernel(
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

    # Store data to output (view operation is just a reshape, data remains the same)
    tl.store(output_ptr + offsets, data, mask=mask)


def view_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from view_kernel_impl")
    if len(args) >= 2:
        input_tensor = args[0]
        shape = args[1] if len(args) == 2 else args[1:]
    elif "input" in kwargs:
        input_tensor = kwargs["input"]
        shape = kwargs.get("shape", kwargs.get("size", ()))
    else:
        input_tensor = args[0] if args else None
        shape = kwargs.get("shape", kwargs.get("size", ()))

    if input_tensor is None:
        raise ValueError("Input tensor is required")

    # Handle shape argument
    if isinstance(shape, (tuple, list)) and len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    elif not isinstance(shape, (tuple, list)):
        shape = (shape,)

    # Convert any tensor shapes to tuple of ints
    if isinstance(shape, torch.Size):
        shape = tuple(shape)
    elif isinstance(shape, (tuple, list)):
        shape = tuple(int(s) if isinstance(s, torch.Tensor) else s for s in shape)

    # Store original device
    original_device = input_tensor.device
    needs_cuda = original_device.type == "cpu"

    # Move to GPU if needed
    if needs_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot process CPU tensor")
        input_tensor = input_tensor.cuda()
    elif original_device.type != "cuda":
        raise RuntimeError(f"Unsupported device type: {original_device.type}")

    # Validate the view operation
    input_numel = input_tensor.numel()

    # Handle -1 in shape (infer dimension)
    shape = list(shape)
    neg_ones = [i for i, s in enumerate(shape) if s == -1]
    if len(neg_ones) > 1:
        raise RuntimeError("Only one dimension can be inferred")
    elif len(neg_ones) == 1:
        known_size = 1
        for s in shape:
            if s != -1:
                known_size *= s
        if input_numel % known_size != 0:
            raise RuntimeError(f"Shape {tuple(shape)} is invalid for input of size {input_numel}")
        shape[neg_ones[0]] = input_numel // known_size

    shape = tuple(shape)
    output_numel = 1
    for s in shape:
        output_numel *= s

    if output_numel != input_numel:
        raise RuntimeError(f"Shape {shape} is invalid for input of size {input_numel}")

    # Create output tensor with new shape
    output = torch.empty(shape, dtype=input_tensor.dtype, device=input_tensor.device)

    # If tensor is empty, return immediately
    if input_numel == 0:
        result = output
        if needs_cuda:
            result = result.cpu()
        return result

    # Launch kernel
    n_elements = input_numel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    view_triton_kernel[grid](
        input_tensor,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Move result back to original device if needed
    if needs_cuda:
        output = output.cpu()

    return output
