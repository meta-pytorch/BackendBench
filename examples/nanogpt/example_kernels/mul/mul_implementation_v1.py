# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def mul_triton_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)

    output_vals = input_vals * other_vals

    tl.store(output_ptr + offsets, output_vals, mask=mask)


def mul_kernel_impl(*args, **kwargs):
    # Handle both positional and keyword arguments
    # print("Hello from mul_kernel_impl")
    if len(args) >= 2:
        input_tensor = args[0]
        other = args[1]
        # Extract any additional kwargs
        out = kwargs.get("out", None)
    elif len(args) == 1:
        input_tensor = args[0]
        other = kwargs.get("other", kwargs.get("input", None))
        if other is None:
            raise ValueError("Missing 'other' argument for multiplication")
        out = kwargs.get("out", None)
    else:
        input_tensor = kwargs.get("input", None)
        other = kwargs.get("other", None)
        if input_tensor is None or other is None:
            raise ValueError("Both 'input' and 'other' arguments are required")
        out = kwargs.get("out", None)

    if input_tensor is None or other is None:
        raise ValueError("Both input tensors are required for multiplication")

    # Store original devices
    input_device = input_tensor.device

    # Check CUDA availability
    if not torch.cuda.is_available():
        if input_tensor.device.type == "cuda" or (
            torch.is_tensor(other) and other.device.type == "cuda"
        ):
            raise RuntimeError("CUDA is not available but GPU tensors were provided")
        # Fallback to CPU implementation
        return torch.mul(input_tensor, other, out=out)

    # Move tensors to GPU if needed
    if input_tensor.device.type == "cpu":
        input_tensor = input_tensor.cuda()

    if torch.is_tensor(other):
        if other.device.type == "cpu":
            other = other.cuda()
    else:
        # Convert scalar to tensor on same device as input
        other = torch.tensor(other, device=input_tensor.device, dtype=input_tensor.dtype)

    # Ensure tensors are broadcastable and get output shape
    try:
        output_shape = torch.broadcast_shapes(input_tensor.shape, other.shape)
    except RuntimeError as e:
        raise ValueError(f"Tensors are not broadcastable: {e}")

    # Broadcast tensors to the same shape
    if input_tensor.shape != output_shape:
        input_tensor = input_tensor.broadcast_to(output_shape)
    if other.shape != output_shape:
        other = other.broadcast_to(output_shape)

    # Ensure contiguous tensors
    input_tensor = input_tensor.contiguous()
    other = other.contiguous()

    # Create output tensor
    if out is not None:
        if out.shape != output_shape:
            raise RuntimeError(
                f"Output tensor shape {out.shape} doesn't match expected shape {output_shape}"
            )
        if out.device.type == "cpu" and torch.cuda.is_available():
            output = out.cuda()
        else:
            output = out
    else:
        output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    n_elements = output.numel()

    if n_elements == 0:
        # Handle empty tensors
        result = output
    else:
        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        mul_triton_kernel[grid](
            input_tensor,
            other,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        result = output

    # Move result back to original device if needed
    target_device = input_device
    if result.device != target_device:
        result = result.to(target_device)

    # Handle out parameter
    if out is not None:
        if out.device != result.device:
            out.copy_(result.to(out.device))
        else:
            out.copy_(result)
        return out

    return result
